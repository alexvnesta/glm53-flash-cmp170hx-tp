"""CPU-only row64 rebuild, including every cooperative-header translation unit."""
import argparse
from pathlib import Path
from build_row16 import build,NATIVE_SUBDIR,REBUILD

UNITS=tuple(NATIVE_SUBDIR/(name+'.cpp') for name in REBUILD)+(
    NATIVE_SUBDIR/'quant/exl3_moe_coop.cu',)+tuple(
    NATIVE_SUBDIR/f'quant/comp_units/exl3_moe_coop_inst_{name}.cu'
    for name in [*(f'k{k}' for k in range(1,9)),*(f'h{k}' for k in range(1,4))])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-profile',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--base-native-sha256',required=True)
    a=p.parse_args()
    build(a.base_profile.resolve(),a.candidate.resolve(),a.python.absolute(),
          a.base_native_sha256,translation_units=UNITS)
