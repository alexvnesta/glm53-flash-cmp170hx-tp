// Experimental KV-only CUDA demand paging. No weights/workspaces use this allocator.
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <cstring>
#include <cstdio>

extern "C" void* kv_managed_malloc(ssize_t size, int device, cudaStream_t stream) {
    (void)stream;
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrConcurrentManagedAccess, device) != cudaSuccess || !supported)
        return nullptr;
    void* ptr = nullptr;
    auto err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "KV cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    // First-touch on CPU, avoiding a full-pool GPU zero pass at allocation.
    // Do not set a preferred GPU: that overrides CUDA's thrash mitigation.
    std::memset(ptr, 0, size);
    return ptr;
}

extern "C" void kv_managed_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    (void)size; (void)device; (void)stream;
    cudaFree(ptr);
}

// Oracle-only migration control. Serving uses automatic demand paging.
extern "C" int kv_prefetch(void* ptr, size_t size, int device, cudaStream_t stream) {
    cudaMemLocation location{};
    location.type = device < 0 ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    location.id = device < 0 ? 0 : device;
    return cudaMemPrefetchAsync(ptr, size, location, 0, stream);
}
