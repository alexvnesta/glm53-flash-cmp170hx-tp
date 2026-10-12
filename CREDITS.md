# Credits

- **Flun**, [glm53-flash-cmp170hx-exl3](https://github.com/Flun/glm53-flash-cmp170hx-exl3): the original two-card
  GLM-5.3-Flash EXL3 server and recipe this work starts from. The OpenAI-compatible server, the 3.05 bpw + DFlash2 K7 + Q8
  recipe, the drafter conversion to EXL3 6 bpw, the drafter's sliding-window GPU ring that makes ~393K context fit, Q8 cache
  staging, lm_head placement, the async generator and the tool-call parser are his. Nine of his files are still unchanged here.
  His layer-split layout ran at 96.2 / 34.2 tok/s (easy / hard text) on our machine; this tensor-parallel build at 123.6 / 44.3.
- **turboderp**, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the
  [GLM-5.3-Flash EXL3 quants](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), including GLM-5.3 tensor parallelism,
  the vision tower support and native DFlash2.
- **Z.ai**, GLM-5.3-Flash.
- **incoai**, the [GLM-5.3-Flash DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
- **0xSero**, the `k_hcfuse` fused hyper-connection kernels (via Flun's repo).
- **MiaAI-Lab**, the DFlash2 EXL3 conversion reference Flun credits.
- **Morrowmake** ([glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe), MIT): the idea of verifying
  GPU peer-to-peer by content before enabling it on these cards, which `scripts/p2p_check.py` reimplements, and the benchmark prompt
  style used for comparison in docs/PERFORMANCE.md.
- **bayley** and **arvkmr** ([vllm-170hx-glm5](https://github.com/bayley/vllm-170hx-glm5)): the all-zero-row fault probe idea and
  the router-precision findings from their vLLM work on these cards. We reimplemented the ideas; no code was copied.
- Flun's r/LocalLLaMA write-up and its commenters, which prompted the comparisons in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
