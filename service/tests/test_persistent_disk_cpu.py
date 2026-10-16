"""Actual upstream disk serialization, restart reuse and corruption handling; GPUs hidden."""
from collections import deque
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch
from exllamav3.cache import Cache
from exllamav3.cache.recurrent import RecurrentCache
from exllamav3.generator.generator import Generator
from exllamav3.generator.disk_cache import DiskPageCache
from exllamav3.modules.gated_delta_net import GDNLayerState


class RealDiskCPU(unittest.TestCase):
    def make(self, root, alpha=2):
        slabs = torch.empty((2, 256, 16), dtype=torch.uint8)
        slabs[0].fill_(71)
        slabs[1].fill_(89)
        tier = NS(slab_size=4096, segments=[(slabs, 0, (256,16), torch.uint8)],
                  tp_groups=[], max_slots=2, slab_view=lambda slot:slabs[slot].view(-1))
        cache = NS(model=NS(config=NS(config_dict={'test':True}, directory=None)), layers={})
        disk = DiskPageCache(root, [cache], tier, 1024**2, min_free=0,
                             extra_identity={'weightless':{'alpha':alpha,'vector_sha256':'a'*64}})
        return disk, slabs

    def test_pages_and_paired_state_survive_restart_but_not_profile_change(self):
        with tempfile.TemporaryDirectory() as root:
            h=hashlib.blake2b(b'page',digest_size=16).digest()
            tokens=torch.arange(256,dtype=torch.long)
            disk, slabs=self.make(root)
            expected=slabs[0].clone()
            self.assertTrue(disk.write_page(h,None,tokens,0,[]))
            disk.flush()
            state={'position':256,'checkpoint_size':32,
                   'target':{'position':256,'state':torch.arange(16,dtype=torch.half)}}
            self.assertTrue(disk.write_state(h,state,force=True))
            disk.flush()
            fingerprint=disk.fingerprint
            disk.close()
            disk, _=self.make(root)
            try:
                self.assertEqual(disk.fingerprint,fingerprint)
                self.assertTrue(disk.has_page(h))
                dest=torch.zeros(4096,dtype=torch.uint8)
                self.assertTrue(torch.equal(disk.read_page_into(h,dest),tokens))
                self.assertTrue(torch.equal(dest.view(256,16),expected))
                restored=disk.read_state(h)
                self.assertTrue(torch.equal(restored['target']['state'],state['target']['state']))
                self.assertEqual(restored['target']['state'].dtype,torch.half)
            finally:disk.close()
            other,_=self.make(root,alpha=1)
            try:
                self.assertNotEqual(other.fingerprint,fingerprint)
                self.assertFalse(other.has_page(h))
            finally:other.close()

    def test_corrupt_page_is_a_miss(self):
        with tempfile.TemporaryDirectory() as root:
            h=hashlib.blake2b(b'corrupt',digest_size=16).digest()
            disk,_=self.make(root)
            disk.write_page(h,None,torch.arange(256),0,[])
            disk.flush()
            p=Path(disk._page_path(h))
            data=bytearray(p.read_bytes());data[-1]^=255;p.write_bytes(data)
            self.assertIsNone(disk.read_page_into(h,torch.empty(4096,dtype=torch.uint8)))
            disk.close()


class NativeRecoveryCPU(unittest.TestCase):
    def test_native_cache_returns_slots_after_constructor_failure(self):
        for name,args in [('get_new_state',()),('get_test_state',(8,)),('new_from_stashed',({},8))]:
            for error in (RuntimeError,MemoryError,KeyboardInterrupt):
                cache=Cache.__new__(Cache);cache.free_list=deque([1,2])
                def fail(*a,**kw):raise error('injected allocation failure')
                cache.recurrent_state_cls=fail
                with self.assertRaises(error):getattr(cache,name)(*args)
                self.assertEqual(list(cache.free_list),[1,2])

    def test_native_pair_frees_target_if_draft_allocation_fails(self):
        g=Generator.__new__(Generator);freed=[]
        target=NS(free=lambda:freed.append('target'))
        def fail():raise MemoryError('draft allocation')
        g.cache=NS(get_new_state=lambda:target);g.draft_cache=NS(get_new_state=fail)
        g.target_recurrent=g.draft_recurrent=True
        with self.assertRaises(MemoryError):g.new_recurrent_state()
        self.assertEqual(freed,['target'])

    def test_oversized_checkpoint_does_not_evict_or_stash(self):
        c=RecurrentCache(NS(),max_size=1)
        c[b'existing']={'checkpoint_size':1}
        def forbidden():raise AssertionError('oversized state must not be copied')
        c.put(b'new',NS(checkpoint_size=2,stash=forbidden))
        self.assertEqual(list(c),[b'existing'])
        self.assertEqual(c.metrics['stash_skipped_budget'],1)

    def test_fp16_changes_storage_and_checkpoint_geometry_without_cuda(self):
        m=NS(fdim_qkv=12,conv_kernel_size=4,num_v_heads=2,k_head_dim=4,v_head_dim=4)
        state=GDNLayerState(m,2,7,123)
        before=state.get_checkpoint_size()
        state.set_state_dtype(torch.half)
        self.assertEqual(state.recurrent_state.device.type,'meta')
        self.assertEqual(state.recurrent_state.dtype,torch.half)
        self.assertEqual(before-state.get_checkpoint_size(),2*4*4*2)


if __name__=='__main__':unittest.main()
