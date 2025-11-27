## Highlights
- Added a full pytest suite (unit + slow integration) and CI configuration so we automatically verify imports, CLI usage, and model loading before every release.
- Documented the new testing workflow in `docs/dev/PRINCIPLES.md` and `docs/dev/MAINTENANCE.md` so future updates stay consistent.
- Added module aliasing in `demucs_infer.compat` to keep pretrained checkpoints working after the package rename, and fixed the CLI parser name to show `demucs-infer` everywhere.
- Simplified packaging by making sure YAML/TXT configs ship with the wheel and adding validation to the publish workflow to catch missing artifacts.

## Testing
- `uv run pytest tests/ -v`
- `uv run python examples/basic_separation.py`
- `uv run demucs-infer --help`
- `uv run demucs-infer --list-models`
