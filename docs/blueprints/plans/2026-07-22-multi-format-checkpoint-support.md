# Multi-format Demucs checkpoint support — plan

> Generated: 2026-07-22 · Spec source: this session's Demucs checkpoint scout and compatibility probes · Stage 1: fresh, reusing the immediately preceding read-only checkpoint audit

## Context

`demucs-infer` currently has three overlapping checkpoint systems. The historical resolver in `demucs_infer/pretrained.py` reads Meta's `remote/files.txt` and bag YAML files; `demucs_infer/community.py` separately owns DrumSep download metadata; and the clean facade in `demucs_infer/clean_api.py` reads the package-owned TOML but reimplements resolution, downloading, cache lookup, and bag materialization. The public `CHECKPOINT_CATALOG` compatibility view is generated from that TOML, but schema v1 flattens every model to one artifact and cannot describe a multi-artifact bag or a raw state-dict construction recipe.

The compatibility probes established three distinct packaging cases over the same existing inference runtime. Both UVR HDemucs checkpoints load unchanged through `states.load_model`. All three CDX23 checkpoints also load unchanged as `HTDemucs`, but their downloaded filenames share the `97d170e1-` prefix and therefore collide in `LocalRepo` even though their payload hashes differ. The MSST vocals checkpoint is a raw `OrderedDict`, so the current loader raises `KeyError: 'klass'`; constructing the existing `HTDemucs` from the published MSST config and loading the state dict with `strict=True` succeeds with all keys matched.

The externally consumed contract is frozen: existing imports and signatures for `get_model`, `Separator`, `DemucsSeparator`/`DemucsSession`, `separate*`, `CHECKPOINT_CATALOG`, `get_checkpoint_metadata`, CLI behavior, explicit `repo=Path(...)`, and custom checkpoint overrides must remain compatible. Model architecture and forward-pass modules are not refactor targets. The intent is to make one package-owned registry and one runtime path own checkpoint recipes, artifact acquisition, cache reporting, and loading, then use that path to support all current official models plus the six newly scouted artifacts.

## Gap analysis

| Domain | Current shape | Target need | Gap / preparation |
|---|---|---|---|
| Registry | TOML schema v1 records six physical weights and only the first file per model | Describe physical artifacts once, named model recipes, aliases, bags, and loader format | Introduce schema v2 with compatibility views; backfill 27 official artifacts and 11 official bags |
| Resolution and cache | `clean_api.py`, `pretrained.py`, and `community.py` each know part of the search/download chain | One read-only resolver shared by load and `cache_info()`, plus one materializer | Extract a checkpoint runtime deep module and rewire every default named-model caller |
| Deserialization | `states.load_model` only accepts Meta export packages containing `klass/args/kwargs/state` | Load Meta/UVR/CDX23 export packages and recipe-driven raw state dicts | Add explicit format dispatch; retain `states.load_model` behavior unchanged for existing callers |
| Bags | Packaged YAML plus copied cache YAML; signatures are inferred from filenames | Declarative component IDs independent of remote filenames | Build bags from recipe components and stop using filename prefixes as model identity |
| Metadata | `model_info.py`, README tables, TOML, `files.txt`, and `COMMUNITY_MODELS` drift independently | Registry-derived listing and accurate source/stem metadata | Rewire metadata/listing; remove phantom entries and correct UVR/CDX23 labels |
| Verification | Existing official baseline and lifecycle tests; no third-party model parity fixtures | Prove old behavior unchanged and each new recipe loads and infers correctly | Capture baselines before structural work, then add offline contract tests and network/real-weight gates |

## Resolved questions

| Question | User's answer |
|---|---|
| Should the work include only the six new artifacts or also repair the incomplete official manifest? | Include the six new artifacts and backfill all 27 already-supported official weights plus 11 official bags. |
| What are the stable user-facing model names? | Use `msst_htdemucs_vocals`, `cdx23_dnr`, `uvr_demucs_model_1`, `uvr_demucs_model_2`, and `uvr_demucs_model_bag`; keep source signatures as advanced aliases where unambiguous. |
| What source policy applies in this change? | Pin the original public sources and SHA-256 values now, design generic multi-URL fallback, and defer openmirlab mirroring until separate outward-facing approval. |
| Are Demucs v1/v2 checkpoints included? | No. The twelve v1/v2 artifacts are deferred until a separate compatibility probe shows whether they need old runtime support. |
| Where should the plan live? | Use the repository's `docs/blueprints/plans/` convention. |

## Approach

### Phase 0 — Freeze behavior and record real-checkpoint baselines

1. Record the current public signatures and exports for the frozen contract.
2. Run the full offline suite and existing bit-identical `htdemucs` baseline before any refactor.
3. Add reproducible probe tools/fixtures for the untouched current loader:
   - UVR Model 1 and Model 2 loaded directly from their `.th` exports.
   - The three CDX23 components loaded directly by path and assembled with the published bag recipe.
   - MSST `HTDemucs` instantiated from the published inference-relevant config and loaded from its raw state dict.
4. Record environment/device metadata with numeric fixtures. Compare discrete metadata exactly; guard floating-point digests to their recorded environment as required by the repository accuracy policy.

Verification: existing tests pass; `tools/capture_baseline.py` remains bit-identical; each probe can be regenerated from pinned source URLs and hashes before production code changes.

### Phase 1 — Behavior-preserving checkpoint-boundary refactor

1. Freeze `states.load_model(path_or_package, strict=False)` as the Meta-export compatibility loader; do not change its public behavior.
2. Introduce a checkpoint runtime deep module with three internal responsibilities behind a narrow interface:
   - resolve a stable model name or alias into an immutable recipe without touching the network;
   - materialize all recipe artifacts atomically into the configured cache and verify full SHA-256 values;
   - load each component by its declared format and assemble a single model or `BagOfModels`.
3. Move the existing download/verification/cache logic out of `clean_api.py` without semantic changes, then rewire `DemucsSeparator.load()` and `cache_info()` to the same resolver. `cache_info()` must remain read-only.
4. Rewire default `pretrained.get_model(name, repo=None)` through the package runtime while preserving explicit `repo=Path(...)` behavior and a legacy resolver fallback during migration.
5. Keep `COMMUNITY_MODELS`, `GDriveRepo`, `CHECKPOINT_CATALOG`, and `get_checkpoint_metadata` as compatibility views/adapters, but remove their authority over duplicated checkpoint facts.

Verification: signature/export diff is unchanged; old unit tests pass; the existing official baseline is bit-identical; a same-domain grep shows resolution, cache lookup, and download policy have one production owner and all intended callers use it.

### Phase 2 — Adopt checkpoint schema v2 as the single source of truth

1. Upgrade the package TOML schema so it separates:
   - physical artifacts: stable ID, cache filename, ordered HTTPS URLs, SHA-256, optional size, provenance, source revision, and update date;
   - model recipes: stable public name, aliases, ordered component artifact IDs, loader format, optional architecture recipe, bag weights/segment, stems, and display metadata.
2. Backfill the 27 existing official artifacts and 11 official bags from `remote/files.txt` and packaged YAML without changing their public model names or outputs.
3. Represent DrumSep in the same registry while preserving `49469ca8` and the existing optional GDrive compatibility surface.
4. Generate compatibility mappings and model-info/listing data from schema v2 rather than copying names, stems, and URLs into Python tables.
5. Validate duplicate aliases, missing artifacts, unsafe/absolute cache paths, unsupported formats/architectures, malformed hashes, and invalid bag references at config parse time with actionable errors.

Verification: package-data tests cover the schema; every official legacy name resolves to the same components and URL as before; malformed-config tests fail closed; wheel inspection confirms the TOML is shipped.

### Phase 3 — Add the six new artifacts and five public recipes

1. Add UVR artifacts `ebf34a2db` (Model 1) and `ebf34a2d` (Model 2), their full SHA-256 values, individual recipes, and `uvr_demucs_model_bag`. Load them with the existing Meta-package loader.
2. Add the three CDX23 DnR artifacts under distinct internal component IDs and cache filenames, independent of their shared remote experiment prefix. Add the `cdx23_dnr` bag recipe with outputs in checkpoint order (`music`, `sfx`, `speech`); user-facing documentation may explain that `speech` is the dialogue stem but must not silently rename the programmatic key.
3. Add `msst_htdemucs_vocals` with loader format `state_dict` and an inference-only `HTDemucs` construction recipe derived from the published config. Store only constructor fields required to reproduce the architecture; do not ship training configuration or generic arbitrary-class import strings.
4. Implement a closed architecture factory for raw state dicts. Initially it accepts only `HTDemucs`; unknown architecture names fail explicitly. Normalize typed manifest values before construction and call `load_state_dict(..., strict=True)`.
5. Make all five names work through `get_model`, `Separator`, `DemucsSession`, one-shot helpers, model listing, and the CLI without requiring callers to provide a local repo.

Verification: real files pass SHA-256 checks and strict loading; source/stem metadata match the loaded models; each new public name completes a short real inference; UVR/CDX23 outputs match Phase 0 direct-load fixtures and MSST matches the Phase 0 config-instantiated fixture within the recorded environment's exact/tolerance rule.

### Phase 4 — Remove drift and complete the public surface

1. Remove runtime dependence on copied cache YAML for registry-backed bags; retain packaged legacy YAML only for explicit legacy-resolution compatibility until its callers are proven absent or intentionally preserved.
2. Correct the existing UVR Model 1/2 inversion, replace the misleading single-entry CDX23 metadata with the real three-component bag, and remove `phantom_center` from the Demucs compatibility table because no Demucs checkpoint/source was verified.
3. Update README, CLAUDE.md, CHANGELOG, model-list output, manual-download instructions, cache location/override documentation, and provenance records together.
4. Inspect the corresponding `openmirlab-skills` capability guidance and update it if Demucs model-selection examples or supported-task routing are affected; otherwise record the no-change check in the handoff.
5. Do not publish, tag, upload mirrors, or create releases in this implementation window.

Verification: documentation names are generated from or checked against registry reality; every advertised name loads; no unsupported phantom entry remains; CLI/listing tests and manual-download examples agree with the package resolver.

### Phase 5 — Release-quality verification

1. Run the full offline suite and the opt-in network liveness suite for every configured primary URL.
2. Re-run the original `htdemucs` bit-identical fixture and all new real-checkpoint parity probes.
3. Build sdist and wheel, install the wheel-from-sdist in a clean environment, import public symbols, validate package data, and exercise one registry lookup without downloading.
4. Run inference-only scans to prove no MSST training/evaluation surface entered the package.
5. Review the final diff for frozen-contract changes, duplicate resolver implementations, stale headers, registry/docs disagreement, and accidental weight files.

Verification: all gates pass with recorded commands/results; the work stops before release/publishing for explicit user sign-off.

## Critical files

| File | Why it matters | Touched in phase |
|---|---|---|
| `demucs_infer/config/checkpoints.toml` | Package-owned artifact and recipe source of truth | 2, 3 |
| `demucs_infer/checkpoint_catalog.py` | Schema parser and legacy read-only catalog views | 2 |
| `demucs_infer/checkpoint_runtime.py` (new) | Unified resolve/materialize/load boundary | 1, 3 |
| `demucs_infer/states.py` | Frozen Meta-export loader reused by format dispatch | 1, 3 |
| `demucs_infer/pretrained.py` | Legacy `get_model` contract and migration fallback | 1, 3 |
| `demucs_infer/clean_api.py` | Public lifecycle facade; delegates checkpoint ownership after refactor | 1 |
| `demucs_infer/community.py` | DrumSep/GDrive backward-compatible adapter | 1, 2 |
| `demucs_infer/model_info.py` | Public listing and metadata, currently carrying stale phantom entries | 2, 4 |
| `demucs_infer/remote/*.yaml` and `remote/files.txt` | Historical official registry used for parity and fallback | 2, 4 |
| `tests/test_checkpoint_config.py` | Schema, package-data, and fail-closed validation | 2 |
| `tests/test_clean_api.py` | Lifecycle, download, cache, and no-reload contract | 1, 3 |
| `tests/test_pretrained_community.py` | Legacy resolver compatibility | 1, 2 |
| `tests/test_checkpoints_liveness.py` | Real configured URL availability | 2, 3, 5 |
| `tests/fixtures/` and `tools/` probe scripts | Before/after accuracy evidence | 0, 3, 5 |
| `README.md`, `CLAUDE.md`, `CHANGELOG.md` | Paired user/maintainer completion gate | 4 |

## Single-source-of-truth owners

| Decision | Owner |
|---|---|
| Stable model names, aliases, stems, format, and bag composition | `demucs_infer/config/checkpoints.toml` model recipes |
| Artifact URLs, cache filenames, hashes, provenance, and sizes | `demucs_infer/config/checkpoints.toml` artifact records |
| TOML parsing and compatibility projections | `demucs_infer/checkpoint_catalog.py` |
| Resolution, cache path selection, atomic materialization, and model assembly | `demucs_infer/checkpoint_runtime.py` |
| Meta export package deserialization semantics | `demucs_infer/states.py::load_model` |
| Raw state-dict architecture construction | Closed factory inside `demucs_infer/checkpoint_runtime.py` |
| Public model descriptions | Registry recipe metadata projected by `demucs_infer/model_info.py` |

Each new owner is adopted in the phase that creates it: Phase 1 rewires clean and legacy default callers to the runtime; Phase 2 rewires Python metadata views to the TOML; Phase 4 verifies no parallel production implementation remains.

## Verification

1. Phase 0 → verify: `uv run pytest tests/`; capture and compare `tests/fixtures/baseline_htdemucs.json`; run each pinned direct-load probe.
2. Phase 1 → verify: public signature/export snapshot unchanged; full offline suite; official baseline bit-identical; grep confirms one resolver/materializer owner.
3. Phase 2 → verify: schema/config tests, official resolver equivalence tests, malformed-config failures, package-data inspection.
4. Phase 3 → verify: full-hash downloads, strict-load tests, short real inference for all five names, and fixture parity for all six artifacts.
5. Phase 4 → verify: CLI/model-list tests, documentation-to-registry consistency, manual-download paths, capability-guide inspection.
6. Phase 5 → verify: `uv run pytest tests/`; `uv run pytest tests/test_checkpoints_liveness.py -m network`; bit-identical legacy baseline; all new parity probes; inference-only scan; sdist → wheel → clean-venv import/package-data smoke.

End-to-end: from a clean cache, load each of `msst_htdemucs_vocals`, `cdx23_dnr`, `uvr_demucs_model_1`, `uvr_demucs_model_2`, and `uvr_demucs_model_bag` through `DemucsSession`, run a deterministic audio fixture, verify advertised stems and parity, release/reload the session, and confirm `cache_info()` reports exactly the artifacts the loader used without network access.

## Out of scope

- Demucs v1/v2 checkpoint support or an old runtime path.
- Any model architecture or forward-pass rewrite.
- Checkpoint conversion scripts.
- openmirlab-hosted checkpoint mirrors until separately approved.
- Phonon integration or checkpoint-quality admission decisions.
- Publishing, tagging, or creating a GitHub/PyPI release.
