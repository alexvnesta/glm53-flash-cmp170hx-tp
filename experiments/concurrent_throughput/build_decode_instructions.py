"""CPU-only incremental K3 rebuild for the isolated instruction experiments."""
import argparse,json
from pathlib import Path
from build_row16 import build
from prepare_decode_instructions import UNIT

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-profile',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--base-native-sha256',required=True)
    a=p.parse_args();assert json.loads((a.candidate/'preparation.json').read_text())['kind'] in ('routed_a_decode_lut','routed_a_vabsdiff','routed_a_decode_l1')
    build(a.base_profile.resolve(),a.candidate.resolve(),a.python.absolute(),a.base_native_sha256,translation_units=(Path(UNIT),))
