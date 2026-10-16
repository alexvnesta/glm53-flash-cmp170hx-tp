// Exhaustive CUDA comparison against the production decoder on both SM80 GPUs.
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#ifdef EXL3_VERIFY_VABSDIFF
#include "exl3_mul1_vabsdiff.cuh"
#else
#include "exl3_mul1_lut.cuh"
#endif

#define CUDA_OK(x) do { auto err=(x); if(err!=cudaSuccess) { std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(err)); return 1; } } while(0)

__global__ void verify(unsigned int* failures)
{
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=65536) return;
    unsigned int product=i*0x83DCD12Du;
    unsigned int sum=__dp4a(product,0x01010101u,0x6400u);
    half h=__ushort_as_half((unsigned short)sum);
    half2 decoded=__hfma2(__half2half2(h),__half2half2(__ushort_as_half(0x1eee)),
                         __half2half2(__ushort_as_half(0xc931)));
#ifdef EXL3_VERIFY_VABSDIFF
    half2 actual=exl3_mul1_vabsdiff::decode_pair(i|0xa5a50000u, (i^0x1234u)|0x5a5a0000u);
    unsigned int product1=(i^0x1234u)*0x83DCD12Du;
    unsigned int sum1=__dp4a(product1,0x01010101u,0x6400u);
    half2 expected1=__hfma2(__half2half2(__ushort_as_half((unsigned short)sum1)),
        __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
    if(__half_as_ushort(__low2half(actual))!=__half_as_ushort(__low2half(decoded)) ||
       __half_as_ushort(__high2half(actual))!=__half_as_ushort(__low2half(expected1))) atomicAdd(failures,1u);
#else
    if(__half_as_ushort(__low2half(decoded))!=__ldg(exl3_mul1_lookup::values+i)) atomicAdd(failures,1u);
#endif
}

int main()
{
    int devices=0; CUDA_OK(cudaGetDeviceCount(&devices)); if(devices!=2) return 2;
    for(int device=0;device<devices;++device) {
        CUDA_OK(cudaSetDevice(device)); cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,device));
        if(prop.major!=8 || prop.minor!=0) return 3;
        unsigned int* failures; CUDA_OK(cudaMalloc(&failures,sizeof(unsigned int)));
        CUDA_OK(cudaMemset(failures,0,sizeof(unsigned int))); verify<<<256,256>>>(failures);
        CUDA_OK(cudaGetLastError()); unsigned int count; CUDA_OK(cudaMemcpy(&count,failures,sizeof(count),cudaMemcpyDeviceToHost));
        CUDA_OK(cudaFree(failures)); std::printf("{\"device\":%d,\"entries\":65536,\"mismatches\":%u}\n",device,count);
        if(count) return 4;
    }
}
