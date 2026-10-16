"""Isolated SM80 three-bit/mul1 gate-up kernel experiments.

Tests the existing staged decoder, an eight-entry prefetch ring or a different
warp layout only in kernel A. Retiling changes the cross-warp K partition;
the other modes preserve it. Accumulation cadence, routing, down kernels,
other bitrates and communication stay unchanged.
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
    if mode not in ("staged","prefetch8","retile"):raise ValueError("Unknown routed kernel mode")
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
    else:
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
    receipt={'status':'unqualified_candidate','kind':{'staged':'routed_a_staging','prefetch8':'routed_a_prefetch8','retile':'routed_a_retile'}[mode],'decode_max_rows':32,
      'source_files':len(hashes),'base_manifest_sha256':sha(manifest),
      'changed_sources':{name:{'before':expected.get(name),'after':hashes[name]} for name in (UNIT,DERIVED)},
      'derived_from':{DERIVED:{'source':HEADER,'sha256':expected[HEADER]}},
      'native_rebuild_required':True,'gpu_tests_run':False,'serving_changed':False}
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument("--mode",choices=("staged","prefetch8","retile"),default="staged")
    a=p.parse_args();print(json.dumps(prepare(a.engine,a.manifest,a.output,a.mode)))
