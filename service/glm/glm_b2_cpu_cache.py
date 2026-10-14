"""Exact B2-coordinator adaptation of the frozen inactive target-only TP tier.

Rank pool code and page-table eviction mechanics remain unchanged, except that
the copies of one page-table allocation are sent as batched rank RPCs in their
original order (GLM_B2_TIER_TRANSFER_BATCH, default 256; 0 restores one RPC per
page copy). This is an uninstalled source candidate. Constructor registration
is lazy and does not reserve or copy every target K/V page into host RAM.
"""
import os
import threading

import torch
from glm_tp_cpu_cache import LazyTPTargetCPUPageCache
from glm_b2_native import B2Retention, StopSharedGeneration

TRANSFER_BATCH_ENV='GLM_B2_TIER_TRANSFER_BATCH'
DEFAULT_TRANSFER_BATCH=256


def transfer_batch_setting(environ=os.environ):
    """Page copies per batched rank RPC during one allocation; 0 restores one RPC per page copy."""
    raw=environ.get(TRANSFER_BATCH_ENV,'').strip()
    if not raw:return DEFAULT_TRANSFER_BATCH
    if not raw.isdigit() or int(raw)>4096:
        raise ValueError(TRANSFER_BATCH_ENV+' must be an integer from 0 through 4096')
    return int(raw)


class B2TargetCPUPageCache(LazyTPTargetCPUPageCache):
    def __init__(self, coordinator, max_size):
        batch=transfer_batch_setting()
        if type(coordinator) is not B2Retention:
            raise ValueError('Target tier needs the exact combined B2 coordinator')
        coordinator.check_owner()
        g=coordinator.generator
        # Shared mode captures the tier as its epoch, so it must be configured later.
        if (not coordinator.attached or coordinator.concurrency._paired_owner is not coordinator
                or coordinator.tier is not None or coordinator.shared is not None
                or g.cpu_page_cache is not None
                or g.pagetable.cpu_tier is not None or g.num_remaining_jobs()
                or g.job_serial or g.pagetable.referenced_pages
                or g.max_batch_size != 2 or g.cache.num_slots < 2):
            raise ValueError('B2 target tier must register on its exact fresh owned coordinator')
        self.coordinator=coordinator
        super().__init__([g.cache],max_size)
        self.transfer_batch=batch

    def _call(self,function,*args):
        self.coordinator.check_owner()
        try:
            return super()._call(function,*args)
        except StopSharedGeneration:
            raise
        except BaseException as error:
            # A send/ack error can mean some ranks already executed. Retain the
            # physical slot/arenas for whole-process teardown, never retry or
            # present partial rank bytes as a successful page transfer.
            self.coordinator._terminal(error)

    def attach(self,pagetable):
        self.coordinator.check_owner()
        g=pagetable.generator
        if (g is not self.coordinator.generator or g.model is not self.model
                or g.cache is not self.cache or g.max_batch_size != 2
                or g.cache.num_slots < 2 or g.enable_defrag is not False
                or g.num_remaining_jobs() or g.job_serial or pagetable.referenced_pages
                or pagetable.cpu_tier is not None or g.cpu_page_cache is not None
                or pagetable.cache is not self.cache or pagetable.max_pages != self.max_pages
                or pagetable is not self.coordinator.concurrency.pagetable
                or pagetable.all_pages is not self.coordinator.concurrency.all_pages
                or len(pagetable.all_pages)!=self.max_pages
                or any(p.pagetable is not pagetable or p.page_index!=i
                       for i,p in enumerate(pagetable.all_pages))):
            raise ValueError('B2 target tier page geometry/owner changed before attach')
        self._generator_owner=g
        self._page_geometry=(pagetable.all_pages,tuple(id(p) for p in pagetable.all_pages))
        self.pagetable=pagetable

    def _check_owner(self):
        c=self.coordinator
        c.check_owner(transfer=True)
        g=self.pagetable.generator if self.pagetable is not None else None
        if (self.closed or c.tier is not self or g is not self._generator_owner
                or g.model is not self.model or g.cache is not self.cache
                or getattr(g.cache,'owner_serial',None)!=self.cache_owner_serial
                or g.enable_defrag is not False or g.max_batch_size!=2
                or g.cache.num_slots<2 or g.cpu_page_cache is not self
                or self.pagetable.cpu_tier is not self or self.pagetable.cache is not self.cache
                or self.cache.max_num_tokens!=self.max_pages*256
                or self.pagetable.max_pages!=self.max_pages
                or self.pagetable.all_pages is not self._page_geometry[0]
                or tuple(id(p) for p in self.pagetable.all_pages)!=self._page_geometry[1]
                or any(p.pagetable is not self.pagetable or p.page_index!=i
                       for i,p in enumerate(self.pagetable.all_pages))):
            c._terminal(ValueError('B2 target tier lost its exact generator/page geometry'))
        thread=threading.get_ident()
        if self.thread is None:self.thread=thread
        elif self.thread!=thread:
            c._terminal(ValueError('B2 target page transfer left its fixed native step thread'))

    def _check_transfer_owner(self):
        if self._batch is None or not self._batch_proven:
            return super()._check_transfer_owner()
        # Batched pages: the full proof ran at the batch's first page and runs again before each send.
        c=self.coordinator
        if c.closed or c.failure is not None:
            raise StopSharedGeneration('B2 paired coordinator is terminal or closed') from c.failure
        if self.closed or self.thread!=threading.get_ident():
            c._terminal(ValueError('B2 target page transfer left its fixed native step thread'))

    def store(self,page,serial,protect=None):
        self._check_transfer_owner()
        pt=self.pagetable
        if (page.pagetable is not pt or type(page.page_index) is not int
                or not 0<=page.page_index<pt.max_pages or pt.all_pages[page.page_index] is not page
                or page.ref_count!=0 or page.kv_position!=256 or page.phash[:8]==bytes(8)
                or pt.unreferenced_pages.get(page.phash) is not page
                or page.sequence.device.type!='cpu' or page.sequence.dtype!=torch.long
                or tuple(page.sequence.shape)!=(1,256)):
            self.coordinator._terminal(ValueError('Only actual complete unreferenced target evictions may spill'))
        # Acquire token metadata before native DMA or host-slot eviction. Failure
        # here leaves the GPU page and existing host entries untouched.
        tokens=page.sequence.clone()
        entry=self.entries.get(page.phash)
        if entry is not None:
            if entry['prev_hash']!=page.prev_hash or not torch.equal(entry['tokens'],tokens):
                self.coordinator._terminal(ValueError('Target page hash has conflicting host metadata'))
            entry['access_serial']=serial;self.metrics['dedup_hits']+=1
            return
        slot=self._new_slot(protect)
        if slot is None:
            self.metrics['skipped_spills']+=1
            return
        self._transfer(True,slot,page.page_index,page.phash)
        self.entries[page.phash]=dict(slot=slot,prev_hash=page.prev_hash,
                                     access_serial=serial,tokens=tokens)
        self.metrics['pushes']+=1

    def fetch(self,phash,page_index,serial):
        self._check_transfer_owner()
        pt=self.pagetable
        if type(page_index) is not int or not 0<=page_index<pt.max_pages:
            self.coordinator._terminal(ValueError('Target restore page index outside owned pool'))
        page=pt.all_pages[page_index]
        if (page.pagetable is not pt or page.page_index!=page_index
                or page.ref_count!=1 or page.phash!=phash or page.kv_position!=0
                or pt.referenced_pages.get(phash) is not page):
            self.coordinator._terminal(ValueError('Target restore destination is not its fresh native page assignment'))
        return super().fetch(phash,page_index,serial)


def attach_b2_target_tier(coordinator,max_size,factory=B2TargetCPUPageCache):
    if type(max_size) is not int or max_size<0:
        raise ValueError('Target CPU byte budget must be a nonnegative integer')
    if not max_size:return None
    coordinator.check_owner()
    if coordinator.tier is not None:
        raise ValueError('B2 coordinator already owns a target tier')
    tier=None
    try:
        tier=factory(coordinator,max_size)
        tier.attach(coordinator.generator.pagetable)
        # Fresh installation contains no await/callback between these metadata
        # writes. Publish the same exact object in all three ownership sites.
        coordinator.tier=tier
        coordinator.generator.cpu_page_cache=coordinator.generator.pagetable.cpu_tier=tier
        coordinator.check_owner()
        return tier
    except StopSharedGeneration:
        # Ambiguous rank execution cannot be rolled back by more RPCs. Whole
        # target model/process teardown is mandatory before any future job.
        raise
    except BaseException as error:
        # Roll back only our exact newly published pointers. Never clear a
        # replacement/foreign owner as part of failed startup cleanup.
        if tier is not None:
            g=coordinator.generator
            if coordinator.tier is tier:coordinator.tier=None
            if g.cpu_page_cache is tier:g.cpu_page_cache=None
            if g.pagetable.cpu_tier is tier:g.pagetable.cpu_tier=None
            try:
                tier.close()
            except BaseException as cleanup_error:
                error.add_note('B2 target tier startup cleanup failed: '+repr(cleanup_error))
        raise
