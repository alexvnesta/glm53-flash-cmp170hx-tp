# Long-input tokenizer backport

`0001` is the tokenizer change from upstream ExLlamaV3
[`7b336ce74dea`](https://github.com/turboderp-org/exllamav3/commit/7b336ce74deae1dd4d3659e969eafdb849743434),
October 17, 2026. It batches long text chunks across the tokenizer backend's
CPU pool, probes chunk correctness and constructs the output tensor directly
with integer dtype. The source and copyright remain upstream's MIT code.

`0002` restores the caller's special-token mode after the one-time chunking
probe. Without it, the first long call with `encode_special_tokens=False`
can encode literal special-token strings as special IDs. Qualification
reproduced this on the deployed GLM tokenizer, including a mixed-input case.
Both failures remain recorded as the unchanged-upstream negative control.

The corrected candidate passes 54 exact-ID cases: fresh long inputs in both
special-token modes, Unicode/CRLF/whitespace, text without newlines, mixed
inputs, BOS/EOS, batch padding/offsets and the actual MMEmbedding alias-splice
path. The benchmark also compares IDs on every timed call using the model's
own Jinja template and a synthetic code/tool prompt.

`qualify.py` executes the complete actual Tokenizer class, its real tokenizers
backend and CPU tensor construction. It reads only tokenizer/config/template
files, bypassing package initialization and the CUDA extension. It uses small
CPU embeddings to check alias replacement, not a vision tower.

```bash
CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=4 python experiments/tokenizer/qualify.py \
  --engine /absolute/pinned-engine \
  --candidate-tokenizer /absolute/candidate/exllamav3/tokenizer/tokenizer.py \
  --model /absolute/GLM-model --output /absolute/receipt.json --repeats 12
```

For the October 19 serving-host measurement, CPU affinity was limited to four
cores, other CPU libraries to one thread, and priority lowered. Paired arm
order alternated. The 12 repeats per arm include tokenization and CPU tensor
construction. Rendering was timed separately. These synthetic CPU timings
do not establish a model decode or whole-request speedup. Full receipts and
the current qualification status are in
[`docs/RECOVERY_HEADROOM_TOKENIZER_2026-10-19.md`](../../docs/RECOVERY_HEADROOM_TOKENIZER_2026-10-19.md).
