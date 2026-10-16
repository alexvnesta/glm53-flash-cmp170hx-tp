"""Upstream CPUPageCache owned by B2, with upstream disk I/O and a fixed RAM ceiling."""
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.model.model_tp_fn import mp_cpu_cache_init


def mp_close_persistent_tier(context):
    # DiskPageCache.close has drained all rank I/O before Generator closes the RAM tier.
    pool = context.get("cpu_page_cache")
    if pool is not None:
        pool.close()
        context["cpu_page_cache"] = None


class B2PersistentCPUPageCache(CPUPageCache):
    persistent_capable = True
    reservation_policy = "upstream_fixed_registered_rank_arenas"

    def __init__(self, coordinator, max_size):
        coordinator.check_owner()
        self.coordinator = coordinator
        self.closed = False
        g = coordinator.generator
        if coordinator.tier is not None or g.cpu_page_cache is not None or g.pagetable.cpu_tier is not None:
            raise ValueError("A cache tier already owns this generator")
        super().__init__([g.cache], max_size)
        self.rank_slot_bytes = tuple(g.model.tp_worker_dispatch_wait_multi(
            list(g.model.active_devices), mp_cpu_cache_init, ([id(g.cache)], 0)))

    def attach(self, pagetable):
        self.coordinator.check_owner()
        if pagetable is not self.coordinator.generator.pagetable or pagetable.cpu_tier is not None:
            raise ValueError("Persistent RAM tier must attach to its exact fresh page table")
        super().attach(pagetable)

    def _check_owner(self):
        if self.closed:
            raise RuntimeError("Persistent RAM tier is closed")
        # Native close runs after AsyncGenerator has joined its iteration task.
        self.coordinator.check_owner(transfer=not self.coordinator.closing)
        if self.coordinator.tier is not self or self.pagetable is not self.coordinator.generator.pagetable:
            self.coordinator._terminal(RuntimeError("Persistent RAM tier owner changed"))

    def store(self, page, serial, protect=None):
        self._check_owner()
        pt = self.pagetable
        if (page.pagetable is not pt or type(page.page_index) is not int
                or not 0 <= page.page_index < pt.max_pages or pt.all_pages[page.page_index] is not page
                or page.kv_position != 256 or page.can_revert):
            self.coordinator._terminal(ValueError("Only settled whole pages can enter the persistent RAM tier"))
        return super().store(page, serial, protect)

    def fetch(self, phash, page_index, serial):
        self._check_owner()
        pt = self.pagetable
        if type(page_index) is not int or not 0 <= page_index < pt.max_pages:
            self.coordinator._terminal(ValueError("Persistent restore index outside owned pool"))
        page = pt.all_pages[page_index]
        if (page.phash != phash or page.ref_count != 1 or page.kv_position != 0
                or pt.referenced_pages.get(phash) is not page):
            self.coordinator._terminal(ValueError("Persistent restore destination is not freshly assigned"))
        return super().fetch(phash, page_index, serial)

    def begin_promote(self, phash, serial, protect=None):
        self._check_owner()
        return super().begin_promote(phash, serial, protect)

    def begin_transfer_batch(self):
        # Upstream dispatches its own rank copies, and its disk workers record/wait on their events.
        return None

    def stats(self):
        size = self.max_slots*self.slot_size
        return {"budget_slots": self.max_slots, "slot_bytes": self.slot_size,
                "rank_slot_bytes": list(self.rank_slot_bytes), "reserved_budget_bytes": size,
                "pinned_bytes": size, "resident_entries": len(self.entries),
                "reservation_policy": self.reservation_policy, "metrics": dict(self.metrics)}

    def close(self):
        if self.closed:
            return
        for model, cache_ids in self.tp_groups:
            model.tp_worker_dispatch_wait_multi(list(model.active_devices), mp_close_persistent_tier, ())
        super().close()
        self.closed = True
