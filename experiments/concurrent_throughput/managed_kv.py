"""KV-only demand-migrated CUDA allocation, kept out of production sources.

CUDA owns migration/eviction; this is not an application LRU policy. One coherent
allocation is accessed through the existing kernels and page-transfer paths.
"""
import ctypes
from pathlib import Path
import torch


class ManagedKV:
    def __init__(self, qshape, sshape, device):
        self.device = torch.device('cuda', device) if isinstance(device, int) else torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('ManagedKV is CUDA-only; preserve CPU export allocation')
        self.allocator = torch.cuda.memory.CUDAPluggableAllocator(
            str(Path(__file__).with_name('managed_kv_allocator.so')),
            'kv_managed_malloc', 'kv_managed_free')
        self.pool = torch.cuda.MemPool(self.allocator.allocator())
        with torch.cuda.device(self.device), torch.cuda.use_mem_pool(self.pool):
            self.q = torch.empty(qshape, dtype=torch.int32, device=self.device)
            self.s = torch.empty(sshape, dtype=torch.float16, device=self.device)
        self.bytes = self.q.numel()*self.q.element_size() + self.s.numel()*self.s.element_size()
        # Validate the allocation kind, rather than inferring paging from a CUDA tag.
        runtime = ctypes.CDLL('libcudart.so.13')
        for tensor in (self.q, self.s):
            attrs = (ctypes.c_ubyte * 128)()
            rc = runtime.cudaPointerGetAttributes(attrs, ctypes.c_void_p(tensor.data_ptr()))
            if rc or ctypes.cast(attrs, ctypes.POINTER(ctypes.c_int))[0] != 3:
                raise RuntimeError('KV allocation is not cudaMemoryTypeManaged')

    def close(self):
        torch.cuda.synchronize(self.device)
        self.q = self.s = None
        # Tensor/cache aliases must be dropped by the caller before this owner.
        self.pool = None
        self.allocator = None
