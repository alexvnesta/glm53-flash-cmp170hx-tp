"""Screen a cache-resident lookup without reducing SM80 block residency.

Only K3/cb2 gate-up uses the lookup. That register path never reads its staging
allocation, so omit those 8 KiB and request a 64 KiB shared/L1 partition (33%
preferred carveout on SM80). Two blocks using about 25 KiB each still fit. Read the fixed
codebook through evict-last loads. The preference is a hint: counters
must verify actual cache configuration and occupancy before qualification.
"""
import argparse,json
from pathlib import Path
from prepare_decode_instructions import prepare as prepare_lookup,ROOT,HEADER,UNIT,DERIVED
from prepare_row16 import sha,verify

INSTANCES=ROOT+'comp_units/exl3_moe_coop_instances.cuh'
GEOMETRY=ROOT+'exl3_moe_coop.cuh'
MACRO='#define EXL3_MOE_COOP_INST_DEF(K) \\\n'
CALL='EXL3_MOE_COOP_INST_DEF(3)'
ANCHOR='        const int smem = smem_a_bytes<3>(Hi);'
CONFIG='''
        if (cb == 2)
        {
            int device; cuda_check(cudaGetDevice(&device));
            static thread_local int configured_device = -1;
            if (configured_device != device)
            {
                int major, minor;
                cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
                cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
                if (major != 8 || minor != 0) throw std::runtime_error("SM80-only lookup experiment");
                cuda_check(cudaFuncSetAttribute(exl3_moe_coop_a_kernel<3, 2, true>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 33));
                cuda_check(cudaFuncSetAttribute(exl3_moe_coop_a_kernel<3, 2, false>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 33));
                configured_device = device;
            }
            void* kernel = wide ? (void*) exl3_moe_coop_a_kernel<3, 2, true> :
                                  (void*) exl3_moe_coop_a_kernel<3, 2, false>;
            return {kernel, smem - smem_stage_bytes<3>()};
        }
'''
LOAD='''
__device__ __forceinline__ unsigned short lookup_keep_l1(uint32_t index)
{
    unsigned short value;
    const unsigned short* ptr = exl3_mul1_lookup::values + (index & 0xffff);
    asm ("ld.global.L1::evict_last.u16 %0, [%1];" : "=h"(value) : "l"(ptr));
    return value;
}
'''


def patch_unit(unit,instances):
    if unit.count(CALL)!=1 or instances.count(MACRO)!=1:
        raise ValueError('Instance anchor changed')
    body=instances.split(MACRO)[1].replace('\\\n','\n').replace('##K','3').replace('<K','<3')
    if body.count(ANCHOR)!=1:raise ValueError('Gate/up shared-memory anchor changed')
    body=body.replace(ANCHOR,ANCHOR+'\n'+CONFIG)
    return '#include <stdexcept>\n'+unit.replace(CALL,body)


def patch_loads(header):
    anchor='__device__ __forceinline__ half2 mul1_lookup_pair'
    if header.count(anchor)!=1:raise ValueError('Lookup helper anchor changed')
    header=header.replace(anchor,LOAD+'\n'+anchor)
    for index in ('x0','x1'):
        old=f'__ldg(exl3_mul1_lookup::values + ({index} & 0xffff))'
        if header.count(old)!=1:raise ValueError('Lookup load anchor changed')
        header=header.replace(old,f'lookup_keep_l1({index})')
    return header


def prepare(engine,manifest,output):
    engine=engine.resolve(strict=True)
    verify(engine,manifest)
    if '#define MOE_COOP_WNT 2 ' not in (engine/GEOMETRY).read_text():
        raise ValueError('Expected two-tile geometry for the 8 KiB staging reservation')
    # Validate both derived patches before creating any output.
    unit=patch_unit((engine/UNIT).read_text(),(engine/INSTANCES).read_text())
    from prepare_decode_instructions import patch_header
    header=patch_loads(patch_header((engine/HEADER).read_text()))
    receipt=prepare_lookup(engine,manifest,output)
    target=output/'engine'
    # Preserve the derived include selected by prepare_lookup.
    unit=unit.replace('#include "../exl3_moe_coop_kernel.cuh"', '#include "../exl3_moe_coop_decode_lut.cuh"')
    (target/UNIT).write_text(unit);(target/DERIVED).write_text(header)
    hashes={
        line.split('  ',1)[1]:line.split('  ',1)[0] for line in (output/'engine.sha256').read_text().splitlines()}
    for name in (UNIT,DERIVED):
        hashes[name]=sha(target/name);receipt['changed_sources'][name]['after']=hashes[name]
    (output/'engine.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    receipt.update(kind='routed_a_decode_l1',preferred_shared_carveout_percent=33,
        unused_gate_up_staging_bytes_removed=8192,requires_sm80=True)
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    verify(target,output/'engine.sha256');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.engine,a.manifest,a.output)))
