"""Prepare isolated SM80 3-bit/mul1 gate/up instruction experiments.

The 16-bit codebook is fixed, independent of weights and routing. Generate its
65536 half bit patterns using exactly representable binary arithmetic; qualify
every entry against the original CUDA half2 FMA before kernel/serving tests.
The lookup uses read-only global memory. The vabsdiff mode instead tests the
older native unsigned byte-sum instruction with unchanged multiply and half2 FMA.
Only the K3 unit includes the derived header; down and all other paths retain
the procedural decoder. No scheduling or accumulation changes are made.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
from prepare_row16 import sha, verify
from prepare_routed_moe import ROOT, HEADER, UNIT, TEMPLATE, CALL

DERIVED = ROOT + 'exl3_moe_coop_decode_lut.cuh'
TABLE = ROOT + 'exl3_mul1_lut.cuh'
DECODE_CALL = '                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);'


def half(bits):
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def codebook_bits(index):
    product = (index * 0x83DCD12D) & 0xffffffff
    byte_sum = sum(product >> shift & 255 for shift in (0, 8, 16, 24))
    # Both binary16 operands and their product/sum are exact in Python binary64.
    # One final binary16 rounding matches __hfma2, subject to exhaustive GPU proof.
    return struct.unpack('<H', struct.pack('<e', half(0x6400 + byte_sum) * half(0x1eee) + half(0xc931)))[0]


def table_header():
    values = [codebook_bits(i) for i in range(65536)]
    rows = [', '.join(f'0x{x:04x}' for x in values[i:i+16]) for i in range(0,65536,16)]
    return ('#pragma once\n// Generated fixed mul1 codebook, 128 KiB per device.\n'
        'namespace exl3_mul1_lookup {\n'
        '__device__ const unsigned short values[65536] = {\n' + ',\n'.join(rows) + '\n};\n}\n')


HELPER = '''
// A read-only global table avoids repeated IMAD/DP4A/HFMA2 decode instructions.
// Keep the same extraction, half bit patterns, fragment order and FOLD cadence.
__device__ __forceinline__ half2 mul1_lookup_pair(uint32_t x0, uint32_t x1)
{
    return __halves2half2(__ushort_as_half(__ldg(exl3_mul1_lookup::values + (x0 & 0xffff))),
                         __ushort_as_half(__ldg(exl3_mul1_lookup::values + (x1 & 0xffff))));
}
__device__ __forceinline__ void dq8_mul1_lookup(uint32_t a, uint32_t b, int s2,
                                               FragB& f0, FragB& f1)
{
    uint32_t w7 = fshift(b, a, s2);
    uint32_t w6 = w7 >> 3, w5 = w7 >> 6, w4 = w7 >> 9;
    uint32_t w3 = fshift(b, a, s2 + 12);
    uint32_t w2 = w3 >> 3, w1 = w3 >> 6, w0 = w3 >> 9;
    f0[0] = mul1_lookup_pair(w0, w1);
    f0[1] = mul1_lookup_pair(w2, w3);
    f1[0] = mul1_lookup_pair(w4, w5);
    f1[1] = mul1_lookup_pair(w6, w7);
}
'''


VABS_TABLE = ROOT + 'exl3_mul1_vabsdiff.cuh'
VABS_DERIVED = ROOT + 'exl3_moe_coop_vabsdiff.cuh'
VABS_SOURCE = r'''#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
namespace exl3_mul1_vabsdiff {
__device__ __forceinline__ half2 decode_pair(uint32_t x0, uint32_t x1)
{
    x0 &= 0xffffu; x1 &= 0xffffu;
    x0 *= 0x83DCD12Du; x1 *= 0x83DCD12Du;
    uint32_t sum0, sum1;
    asm ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(sum0) : "r"(x0), "r"(0), "r"(0x6400u));
    asm ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(sum1) : "r"(x1), "r"(0), "r"(0x6400u));
    return __hfma2(__halves2half2(__ushort_as_half((uint16_t)sum0), __ushort_as_half((uint16_t)sum1)),
        __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
}
}
'''


def patch_header(source, mode='lookup'):
    if mode not in ('lookup','vabsdiff'): raise ValueError('Unknown instruction mode')
    header = 'exl3_mul1_lut.cuh' if mode=='lookup' else 'exl3_mul1_vabsdiff.cuh'
    helper = HELPER if mode=='lookup' else HELPER[HELPER.index('__device__ __forceinline__ void'):].replace('mul1_lookup_pair','exl3_mul1_vabsdiff::decode_pair').replace('dq8_mul1_lookup','dq8_mul1_vabsdiff')
    for anchor in (TEMPLATE, CALL, DECODE_CALL, 'namespace exl3_moe_coop_ns {'):
        if source.count(anchor) != 1:
            raise ValueError('Lookup kernel anchor changed')
    source = source.replace('namespace exl3_moe_coop_ns {', f'#include "{header}"\n\nnamespace exl3_moe_coop_ns {{\n' + helper)
    source = source.replace(TEMPLATE,TEMPLATE.replace('HALF = false','HALF = false, bool GATE_UP = false'))
    source = source.replace(CALL,CALL.replace('WIDE, HALF>','WIDE, HALF, true>'))
    result = source.replace(DECODE_CALL, '''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
                    if constexpr (GATE_UP && cb == 2 && !HALF)
                        dq8_mul1_lookup(awv, bwv, x_s2, f0, f1);
                    else
#endif
''' + DECODE_CALL)
    return result if mode=='lookup' else result.replace('dq8_mul1_lookup(awv','dq8_mul1_vabsdiff(awv')


def prepare(engine, manifest, output, mode='lookup'):
    if mode not in ('lookup','vabsdiff'): raise ValueError('Unknown instruction mode')
    derived_name, table_name = (DERIVED,TABLE) if mode=='lookup' else (VABS_DERIVED,VABS_TABLE)
    engine=engine.resolve(strict=True); output=output.resolve()
    if output.exists() or output==engine or engine in output.parents:
        raise ValueError('Destination must be new and outside the qualified engine')
    expected=verify(engine,manifest)
    if '#define MAX_BSZN 32 ' not in (engine/'exllamav3/exllamav3_ext/libtorch/mlp.h').read_text():
        raise ValueError('Qualified row32 required')
    if any(name in expected for name in (derived_name,table_name)):
        raise ValueError('Lookup sources already present')
    derived=patch_header((engine/HEADER).read_text(),mode)
    unit=(engine/UNIT).read_text(); anchor='#include "../exl3_moe_coop_kernel.cuh"'
    if unit.count(anchor)!=1: raise ValueError('K3 translation-unit anchor changed')
    output.mkdir(parents=True); target=output/'engine'
    shutil.copytree(engine,target,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    (target/derived_name).write_text(derived); (target/table_name).write_text(table_header() if mode=='lookup' else VABS_SOURCE)
    (target/UNIT).write_text(unit.replace(anchor,f'#include "../{Path(derived_name).name}"'))
    hashes={name:sha(target/name) for name in [*expected,derived_name,table_name]}
    (output/'engine.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    receipt={'status':'unqualified_candidate','kind':'routed_a_decode_lut' if mode=='lookup' else 'routed_a_vabsdiff','decode_max_rows':32,
        'source_files':len(hashes),'base_manifest_sha256':sha(manifest),'table_bytes':131072 if mode=='lookup' else 0,
        'changed_sources':{name:{'before':expected.get(name),'after':hashes[name]} for name in (UNIT,derived_name,table_name)},
        'derived_from':{derived_name:{'source':HEADER,'sha256':expected[HEADER]}},
        'native_rebuild_required':True,'gpu_tests_run':False,'serving_changed':False}
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n'); return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--mode',choices=('lookup','vabsdiff'),default='lookup'); a=p.parse_args()
    print(json.dumps(prepare(a.engine,a.manifest,a.output,a.mode)))
