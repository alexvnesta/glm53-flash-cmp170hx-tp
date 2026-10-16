"""Build only the separately attested K3 translation unit, with GPUs hidden."""
import argparse,json
from pathlib import Path
from build_row16 import build
from prepare_routed_moe import UNIT
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-profile',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--python',type=Path,required=True);p.add_argument('--base-native-sha256',required=True)
    a=p.parse_args();receipt=json.loads((a.candidate/'preparation.json').read_text())
    assert receipt['kind'] in ('routed_a_staging','routed_a_prefetch8','routed_a_retile','routed_a_fp32')
    build(a.base_profile.resolve(),a.candidate.resolve(),a.python.absolute(),a.base_native_sha256,translation_units=(Path(UNIT),))
