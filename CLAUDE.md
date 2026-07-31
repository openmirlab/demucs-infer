# demucs-infer

Inference-only fork of Meta's Demucs (music source separation). Training
code has been stripped; this package only loads pretrained checkpoints and
runs separation.

## Status

Beta, actively maintained (`Development Status :: 4 - Beta` in
`pyproject.toml`; current version in `demucs_infer/__about__.py`). Published
to PyPI as `demucs-infer`; GitHub Actions gates every release on the test
suite (`.github/workflows/publish.yml`) -- nothing publishes without it
passing. Upstream `facebookresearch/demucs` is no longer actively
maintained; this fork exists specifically to keep pretrained-model inference
working on current PyTorch/torchaudio. The package-owned schema-v2 registry
also exposes verified compatible Demucs checkpoints without changing model
architectures or the inference algorithms.

## Testing philosophy

Two tiers, both pytest-based:

1. **Accuracy-preservation gate** (below): any change to `demucs_infer/`
   must reproduce the pinned baseline fixture bit-for-bit. This is the
   overriding constraint -- it exists because this package's entire value
   proposition is "same models, same weights, same output," not "improved
   models."
2. **Fast unit/regression suite** (`tests/`): runs by default, network tests
   deselected (`-m "not network"` in `pyproject.toml`'s `addopts`). A
   separate `network` marker covers every configured checkpoint source and is
   opt-in (`pytest -m network`) since it depends on external infrastructure.

## Scope

- `demucs_infer/` -- the package. Model architecture files (`demucs.py`,
  `hdemucs.py`, `htdemucs.py`, `transformer.py`) are deep modules: their
  internals are intentionally left untouched by structural refactors,
  because forward-pass code layout is tied to how each pretrained
  checkpoint's `args`/`kwargs` were pickled (`states.capture_init`).
  Restructuring them risks breaking old-checkpoint loading silently.
- `wdemucs.py` is a deliberate 9-line alias (`WDemucs = HDemucs`) kept for
  unpickling old checkpoints that reference `demucs.wdemucs.WDemucs` by
  class path. Do not remove or merge it into hdemucs.py.
- `compat.py` registers `sys.modules['demucs'] = sys.modules['demucs_infer']`
  (and per-submodule aliases) as an import-time side effect, so old
  checkpoints pickled under the original `demucs.*` module path still
  unpickle. `__init__.py` and `separate.py` both import it first, before
  anything else -- that ordering is load-bearing.
- `demucs_infer/wiener.py` is vendored from open-unmix-pytorch (MIT); see
  its header for attribution and which forward paths actually exercise it
  (htdemucs never does; some mdx/mdx_extra bag members do).
- `demucs_infer/config/checkpoints.toml` is the release-pinned source of truth
  for physical artifacts and named recipes. `checkpoint_runtime.py` alone owns
  named-model resolution, cache paths, verified downloads, format dispatch,
  and registry-backed bag assembly. Packaged `remote/*.yaml` files remain only
  for explicit legacy-repository compatibility; new recipes must not add
  copied YAML files. The default named-model cache is
  `~/.cache/demucs-infer/`; public lifecycle helpers override it with
  `cache_dir`, and README.md owns the exact manual-download table.
- Public compatible model names and exact stems are:
  `uvr_demucs_model_1` (`vocals`, `non_vocals`),
  `uvr_demucs_model_2` (`vocals`, `non_vocals`),
  `uvr_demucs_model_bag` (`vocals`, `non_vocals`), `cdx23_dnr` (`music`,
  `sfx`, `speech`), and `msst_htdemucs_vocals` (`vocals`, `other`). DrumSep
  retains its actual programmatic keys: `bombo`, `redoblante`, `platillos`,
  `toms`.
- `tools/` -- one-off/repeatable scripts (baseline capture, checkpoint
  provenance) that support verification, not part of the installed package.
- `tests/` -- fast unit/regression tests run by default, plus a `network`
  marker for tests that hit real URLs and a `realweights` marker for tests
  needing a real downloaded checkpoint (both deselected by default).

## Backends (Torch / MLX)

Optional, additive `backend=` axis on `Separator`/`DemucsSeparator`/
`DemucsSession`: `None`/`"torch"` (default, unchanged), `"mlx"` (Apple
Silicon, needs the `[mlx]` extra), or `"auto"`. `device` keeps its existing
Torch meaning regardless of `backend` -- it is not overloaded to mean "Apple
Silicon"; `backend="mlx"` only accepts `None`/`"auto"`/`"mps"` for `device`.
See README's "Backends (Torch / MLX)" for the public contract.

- `demucs_infer/backends/` -- the compute seam. `base.py` holds the
  `SeparationBackend` protocol and `BackendUnavailable`; `torch_backend.py`
  is `Separator.separate_tensor()`'s pre-existing body (resample, reference-
  normalize, `apply.apply_model()`, denormalize) moved behind the seam
  verbatim -- a wrapper, never a second implementation. `mlx_backend.py` is
  the new MLX path; its `_apply_chunk` mirrors `apply_model()`'s own
  shift/split/segment recursion (including `apply.TensorChunk`'s exact
  centered-padding-with-real-neighbouring-context semantics for the last,
  short chunk -- a naive "zero-pad this chunk's own edges" implementation
  would silently disagree with Torch on every track's final chunk). Both
  backends are built from an *already-loaded* Torch model
  (`Separator._load_model()` keeps sole ownership of checkpoint resolution,
  sha256 verification, and `BagOfModels` assembly); `mlx_backend.py` never
  touches a checkpoint path or `checkpoint_runtime.py`'s resolver directly.
  `__init__.py` resolves a backend by name. Backend modules import lazily,
  so `import demucs_infer` never pulls in `mlx` -- `tests/test_backends.py`
  asserts that via a subprocess.
- `demucs_infer/mlx/` -- the vendored MLX `HDemucs`/`HTDemucs` (MIT, from
  `ssmall256/mlx-audio-separator`, source revision recorded in file
  headers), imported only by `backends/mlx_backend.py`. Only `HDemucs`/
  `HTDemucs` are ported (not the original time-domain-only `Demucs`, which
  never appears as a single-component registry entry); the Wiener-filtering
  path is not ported (measured unreachable for every checkpoint this
  backend supports -- see `mlx/hdemucs.py`'s module docstring).
  `convert.py`'s `load_converted_weights()` raises rather than loading
  partially -- the naive `model.load_weights(strict=False)` it replaces
  silently drops unmatched keys. `rfft_guard.py`'s `exact_zero_safe_rfft()`
  is applied unconditionally (org policy) but was measured **inert** for
  `htdemucs` (1.937e-07 with the guard vs. 2.533e-07 without, on a
  zero-padded-tail fixture -- see its docstring); this is *not* assumed from
  either sibling package's own finding.
  MLX support is **refused explicitly**, never silently downgraded, for a
  `BagOfModels` of 2+ sub-models (a length-1 bag -- what most single-
  checkpoint registry entries actually resolve to via
  `checkpoint_runtime.load_registered_model()` -- is unwrapped and runs
  normally, since it is mathematically identical to its one submodel), an
  unported architecture, or a Wiener-filtering configuration.
- `.refs/` -- gitignored scratch checkout of `mlx-audio-separator` used only
  to derive/verify the vendored files above; never imported, never shipped.

## Overriding rule: accuracy cannot drop

Any change to `demucs_infer/` must reproduce
`tests/fixtures/baseline_htdemucs.json` bit-for-bit. That fixture was
captured with `random.seed(1234)` pinned immediately before
`apply_model()` (its `shifts` trick draws from the unseeded stdlib
`random` module, not torch's RNG -- see apply.py's header). Re-verify
after every change:

```bash
.venv/bin/python3 tools/capture_baseline.py --out /tmp/baseline_check.json
python3 -c "
import json
a = json.load(open('tests/fixtures/baseline_htdemucs.json'))
b = json.load(open('/tmp/baseline_check.json'))
a['meta'].pop('captured_at'); b['meta'].pop('captured_at')
assert a == b, 'baseline drifted!'
print('bit-identical')
"
```

## Verification commands

```bash
# fast suite (default: network tests deselected)
uv run pytest tests/

# frozen public API and legacy metadata contract
uv run python tools/capture_public_contract.py --compare tests/fixtures/public_contract.json

# verified real-file parity (requires the Phase 0 files in /tmp)
DEMUCS_PHASE0_PROBE_CACHE=/tmp uv run pytest tests/test_phase0_contract.py tests/test_phase3_checkpoints.py -q

# checkpoint URL liveness (hits real network endpoints)
uv run pytest tests/test_checkpoints_liveness.py -m network

# rebuild registry provenance; separately reports recorded hashes and cached files verified
uv run python tools/build_checkpoints_provenance.py

# regenerate the wiener vendoring regression fixture (only after an
# intentional, verified change to demucs_infer/wiener.py)
uv run python tools/capture_wiener_fixture.py
```

## File-top header convention

Load-bearing files (>~150 lines, or files whose purpose isn't obvious from
their name/location) carry a file-top header:

```python
"""<Title -- one line>

<2-3 sentences: what this file does, key design decisions, gotchas a
reader would otherwise discover the hard way.>

Reads: <other demucs_infer modules this file imports and why, or
"(nothing internal)">
"""
```

Thin files (small utilities, `__main__.py`-style entry-point shims) are
deliberately left without a full header when their name and a one-line
docstring already say everything -- don't force the convention there.

## Known, deliberately unfixed issues

- `apply_model(shifts=1)` (the default) draws its random time-shift offset
  from the unseeded stdlib `random` module. Reproducibility across runs
  requires callers to `random.seed(...)` immediately before calling it.
  Not changed: switching to a seeded/local RNG would change every existing
  caller's output distribution (still "valid" outputs, but not the same
  bytes), which conflicts with this repo's own accuracy-preservation bar.
  Recommendation: document loudly (done, see apply.py's header) rather
  than silently reseed on callers' behalf.
- Passing an explicit `segment=` to `apply_model` crashes `HTDemucs` models
  whose `use_train_segment=True` (`ValueError` in `htdemucs.py`'s
  `valid_length`). This is an upstream bug, not introduced by this repo.
  Not fixed: no caller in this codebase or its known downstream consumers
  passes `segment=` explicitly to an HTDemucs model, so there's no
  behavior-preserving way to test a fix without picking an arbitrary
  semantics for the conflict. Recommendation: fix in a dedicated,
  narrowly-scoped change with its own before/after behavioral tests, not
  bundled into an unrelated refactor.
