"""Isolated SM80 three-bit/mul1 gate-up kernel experiments.

Tests staged decode, deeper prefetch, warp layout or FP32 Tensor Core
accumulation only in kernel A. Retiling changes the cross-warp K partition;
FP32 accumulation removes periodic FP16-to-FP32 folding. Other modes preserve
accumulation. Routing, down kernels, other bitrates and communication stay unchanged.
The original header is preserved; only the K3 translation unit includes a new
attested derived header, allowing a one-object incremental rebuild.
"""
import argparse
import json
from pathlib import Path
import shutil
from prepare_row16 import sha,verify

ROOT='exllamav3/exllamav3_ext/quant/'
HEADER=ROOT+'exl3_moe_coop_kernel.cuh'
DERIVED=ROOT+'exl3_moe_coop_routed.cuh'
UNIT=ROOT+'comp_units/exl3_moe_coop_inst_k3.cu'
TEMPLATE='template <int bits, int cb, bool WIDE, bool HALF = false>\n__device__ __forceinline__ void gemv_tile'
REG='    constexpr bool REG = HALF ? false : tile_reg<bits>();'
CALL='gemv_tile<bits, cb, WIDE, HALF>(B32, A2, a_stride2, sh_rows, nrows, C,'


def patch_header(source, mode="staged"):
    if mode not in ("staged","prefetch8","retile","fp32"):raise ValueError("Unknown routed kernel mode")
    for anchor in (TEMPLATE,REG,CALL):
        if source.count(anchor)!=1:raise ValueError('Routed kernel anchor changed')
    source=source.replace(TEMPLATE,TEMPLATE.replace('HALF = false','HALF = false, bool GATE_UP = false'))
    if mode=="staged":
        source=source.replace(REG,'''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    // Test the existing shared-memory decoder for the hot SM80 routed gate/up path.
    constexpr bool REG = HALF ? false : (GATE_UP && bits == 3 && cb == 2 ? false : tile_reg<bits>());
#else
'''+REG+'''\n#endif''')
    elif mode=="prefetch8":
        source=source.replace(REG,REG+'''
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    constexpr int PF = GATE_UP && bits == 3 && cb == 2 ? 8 : exl3_moe_coop_ns::PF;
#endif''')
    elif mode=="retile":
        anchors = (
            '    constexpr int TWORDS = HALF ?',
            '    constexpr int WN = tile_wn<WIDE>();',
            '    constexpr int WKK = tile_wk<WIDE>();',
        )
        for anchor in anchors:
            if source.count(anchor)!=1:raise ValueError('Retile anchor changed')
        # Keep the block's 128 output columns and 512 threads. Only the hot
        # SM80 K3/mul1 wide A path changes: 8 N warps x 2 K warps, one
        # 16-column tile per warp instead of 4 x 4 and two tiles per warp.
        # Scratch remains conservatively sized by the original header.
        source=source.replace(anchors[0],'''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    constexpr bool RETILE = GATE_UP && bits == 3 && cb == 2 && WIDE && !HALF;
#else
    constexpr bool RETILE = false;
#endif
    constexpr int WNT = RETILE ? 1 : exl3_moe_coop_ns::WNT;
    constexpr int COLS = WNT * 16;
'''+anchors[0])
        source=source.replace(anchors[1],'    constexpr int WN = RETILE ? 8 : tile_wn<WIDE>();')
        source=source.replace(anchors[2],'    constexpr int WKK = RETILE ? 2 : tile_wk<WIDE>();')
    else:
        anchors = (
            '    FragC_h ch[WNT][2] = {};',
            '''                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);''',
            '            if ((d + 1) % FOLD == 0 || i + 1 == myn)',
            '''                red[col + 0] = acc0[t][f].x;
                red[col + 1] = acc0[t][f].y;''',
        )
        for anchor in anchors:
            if source.count(anchor)!=1:raise ValueError('FP32 accumulation anchor changed')
        source=source.replace(REG,REG+'''
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    constexpr bool FP32_ACCUM = GATE_UP && bits == 3 && cb == 2 && WIDE && !HALF;
#else
    constexpr bool FP32_ACCUM = false;
#endif''')
        source=source.replace(anchors[0],anchors[0]+'\n    FragC cf[WNT][2] = {};')
        source=source.replace(anchors[1],'''                if constexpr (FP32_ACCUM)
                {
                    // Reuse upstream's FP16-input / FP32-accumulator Tensor Core helper.
                    // The upper eight A rows stay zero, as in the original FP16 path.
                    FragA af;
                    af[0] = a01[0]; af[1] = a01[1];
                    af[2] = a23[0]; af[3] = a23[1];
                    ptx_mma_m16n8k16(af, f0, cf[t][0]);
                    ptx_mma_m16n8k16(af, f1, cf[t][1]);
                }
                else
                {
'''+anchors[1]+'''
                }''')
        source=source.replace(anchors[2],'            if constexpr (!FP32_ACCUM)\n'+anchors[2])
        source=source.replace(anchors[3],'''                if constexpr (FP32_ACCUM)
                {
                    red[col + 0] = cf[t][f][0];
                    red[col + 1] = cf[t][f][1];
                }
                else
                {
'''+anchors[3]+'''
                }''')
    return source.replace(CALL,CALL.replace('WIDE, HALF>','WIDE, HALF, true>'))


def prepare(engine,manifest,output,mode="staged"):
    engine=engine.resolve(strict=True);output=output.resolve()
    if output.exists() or output==engine or engine in output.parents:raise ValueError('Destination must be new and outside the qualified engine')
    expected=verify(engine,manifest)
    if '#define MAX_BSZN 32 ' not in (engine/'exllamav3/exllamav3_ext/libtorch/mlp.h').read_text():raise ValueError('Qualified row32 required')
    if DERIVED in expected:raise ValueError('Derived header already present')
    derived=patch_header((engine/HEADER).read_text(),mode)
    unit=(engine/UNIT).read_text();anchor='#include "../exl3_moe_coop_kernel.cuh"'
    if unit.count(anchor)!=1:raise ValueError('K3 translation-unit anchor changed')
    output.mkdir(parents=True);target=output/'engine'
    shutil.copytree(engine,target,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    (target/DERIVED).write_text(derived)
    (target/UNIT).write_text(unit.replace(anchor,'#include "../exl3_moe_coop_routed.cuh"'))
    hashes={name:sha(target/name) for name in [*expected,DERIVED]}
    (output/'engine.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    receipt={'status':'unqualified_candidate','kind':{'staged':'routed_a_staging','prefetch8':'routed_a_prefetch8','retile':'routed_a_retile','fp32':'routed_a_fp32'}[mode],'decode_max_rows':32,
      'source_files':len(hashes),'base_manifest_sha256':sha(manifest),
      'changed_sources':{name:{'before':expected.get(name),'after':hashes[name]} for name in (UNIT,DERIVED)},
      'derived_from':{DERIVED:{'source':HEADER,'sha256':expected[HEADER]}},
      'native_rebuild_required':True,'gpu_tests_run':False,'serving_changed':False}
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument("--mode",choices=("staged","prefetch8","retile","fp32"),default="staged")
    a=p.parse_args();print(json.dumps(prepare(a.engine,a.manifest,a.output,a.mode)))
