# Project Notes

## Session summary (June 2026)

Starting from the original repo, two rounds of inference speed optimizations
were applied, the README was fully rewritten, and the project was prepared for
public release on the `yossi-optim` branch (since merged to `main`).

---

## Inference Speed Optimizations

Nine optimizations were made across two rounds. The core insight is that the
original code reloaded models and re-read audio from disk on **every file**,
and ran an O(n) Python loop for the DP inner loop.

### Round 1 — Model loading and structural bottlenecks

| # | What | File(s) | Impact |
|---|---|---|---|
| 1 | MMS model loaded once before file loop (was: twice per file) | `inference/models/mms/mms.py`, `align_wav.py` | 2N → 1 model loads |
| 2 | UnsupSeg model loaded once before file loop (was: once per file); added missing `torch.no_grad()` | `inference/models/unsupSeg/unsupseg_classifier.py`, `align_wav.py` | N → 1 model loads |
| 3 | DP distance loop vectorized: one batched call instead of O(n) Python loop (valid for `max_tolerence=1`) | `inference/models/dp_algorithm/utils.py` | ~500 Python→Torch round trips eliminated per utterance |
| 4 | `uroman` singleton cached at module level (was: instantiated twice per file) | `inference/models/utils.py` | 2N → 1 instantiations |

### Round 2 — Remaining bottlenecks

| # | What | File(s) | Impact |
|---|---|---|---|
| 5 | Waveform loaded once per file (was: twice — one read per MMS call path) | `inference/models/preprocess.py`, `mms/mms.py`, `predict.py` | 1 disk read + decode eliminated per file |
| 6 | Removed `torch.cuda.empty_cache()` called inside the hot path (was: one CUDA sync per file, achieves nothing during stable inference) | `inference/models/mms/mms.py` | 1 CUDA sync eliminated per file |
| 7 | DP hot loop: replaced per-frame `torch.tensor(list)` + `.sum()` with `run_functions_weighted()` that accumulates a Python float; pre-built shared `args_for_features` dict; `Features_DP` and `w_floats` pre-instantiated once | `dp_algorithm/extract_features.py`, `dp_algorithm/utils.py`, `align_wav.py` | ~290K tensor allocations eliminated per 60s file |
| 8 | `torch.tensor(np.array(...))` → `torch.as_tensor()` (zero-copy); `torch.no_grad()` → `torch.inference_mode()`; removed redundant `model.eval()` | `inference/models/predict.py` | 2 tensor copies eliminated per file |
| 9 | Removed unused imports (`IPython`, `matplotlib`, `argparse`, `os`, `numpy`) and dead functions (`get_device`, `list_wav_files`) | `mms/mms.py`, `unsupSeg/unsupseg_classifier.py` | Faster module import |

### Summary table

| Operation | Before | After |
|---|---|---|
| MMS model loads | 2N | 1 |
| UnsupSeg model loads | N | 1 |
| Waveform disk reads | 2N | N |
| CUDA `empty_cache` syncs | 2N | 0 |
| DP distance function calls | N × n | N × 1 |
| DP score tensor allocations | N × (eligible frames) | 0 |
| `Features_DP` instantiations | N | 1 |
| Uroman instantiations | 2N | 1 |
| Tensor copies in Conformer | 2 per file | 0 |

*(N = number of files, n = frames per file)*

---

## Other changes

- **`README.md`** — fully rewritten: badges, architecture diagram, 1000+ language
  support, MMS LID note, Interspeech 2026 paper citation with BibTeX, NSF/BSF
  acknowledgements, running examples, argument table.
- **`LICENSE`** — CC BY-NC 4.0 (mirrors upstream MMS-FA license; commercial use
  requires a separate Meta license).
- **`CONTRIBUTORS.md`** — lists all four paper authors.
- **Bug fix** — `torch.load(..., weights_only=False)` added for PyTorch 2.6+
  compatibility (`unsupseg_classifier.py`).

---

## Pending / future ideas

- Add a Gradio demo on HuggingFace Spaces for zero-install public access.
- Submit to [Papers With Code](https://paperswithcode.com) under the arXiv ID `2606.10675`.
- Add GitHub topics: `forced-alignment`, `speech`, `multilingual`, `pytorch`, `word-alignment`.
- Consider a GitHub Actions CI workflow to run the bundled examples on push.
