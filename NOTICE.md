# Third-party notices

| part of this repo | origin | licence |
|---|---|---|
| `service/` | Derived from [Flun/glm53-flash-cmp170hx-exl3](https://github.com/Flun/glm53-flash-cmp170hx-exl3) at commit `5933112`, Copyright (c) 2026 Flun. Flun's licence text is kept in `service/LICENSE`. Our extensions are under this repo's MIT licence | MIT |
| `service/kernels/k_hcfuse.py` | 0xSero's fused hyper-connection decode kernels ([glm53-flash-offload](https://github.com/0xSero/glm53-flash-offload), now `sybil-solutions/glm-flash-lite`), shipped via Flun's repo | MIT |
| `engine/patches/` | Modifications of [ExLlamaV3](https://github.com/turboderp-org/exllamav3), Copyright (c) turboderp. `apply_engine.sh` downloads the upstream source; the patches are distributed under ExLlamaV3's MIT licence | MIT |

Not included and not redistributed:
- **GLM-5.3-Flash** weights (Z.ai, MIT) and the **EXL3 quantization** (turboderp, MIT): downloaded from Hugging Face by
  `scripts/get_models.sh`.
- **DFlash2 drafter** ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)): CC-BY-NC-ND-4.0,
  non-commercial use only, no redistribution of adapted versions. `get_models.sh` converts it locally; do not share the
  converted weights.
