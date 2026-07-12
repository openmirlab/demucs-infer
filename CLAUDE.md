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
working on current PyTorch/torchaudio, not to extend the model family.

## Testing philosophy

Two tiers, both pytest-based:

1. **Accuracy-preservation gate** (below): any change to `demucs_infer/`
   must reproduce the pinned baseline fixture bit-for-bit. This is the
   overriding constraint -- it exists because this package's entire value
   proposition is "same models, same weights, same output," not "improved
   models."
2. **Fast unit/regression suite** (`tests/`): runs by default, network tests
   deselected (`-m "not network"` in `pyproject.toml`'s `addopts`). A
   separate `network` marker covers checkpoint URL liveness against Meta's
   real CDN and is opt-in (`pytest -m network`) since it depends on
   external infrastructure being reachable.

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
- `tools/` -- one-off/repeatable scripts (baseline capture, checkpoint
  provenance) that support verification, not part of the installed package.
- `tests/` -- fast unit/regression tests run by default, plus a `network`
  marker for tests that hit real URLs (deselected by default).

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

# checkpoint URL liveness (hits real network endpoints)
uv run pytest tests/test_checkpoints_liveness.py -m network

# rebuild the checkpoints provenance record (hashes only what's cached locally)
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
