"""Capture the Phase 0 public API, registry, and CLI contract snapshot.

The snapshot records signatures and discrete behavior only; environment-driven
device defaults are normalized to their documented policy. Later checkpoint
refactors compare against this artifact before claiming compatibility.

Reads: demucs_infer public modules, CLI parser, package checkpoint catalog.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
from pathlib import Path

import demucs_infer
from demucs_infer import clean_api
from demucs_infer.api import Separator
from demucs_infer.checkpoint_catalog import CHECKPOINT_CATALOG, get_checkpoint_metadata
from demucs_infer.pretrained import get_model
from demucs_infer.separate import get_parser, main as cli_main
from demucs_infer.states import load_model


AUTO_DEVICE_DEFAULT = "<cuda-if-available-else-cpu>"


def normalized_signature(callable_object, *, dynamic_device: bool = False) -> str:
    signature = inspect.signature(callable_object)
    parameters = list(signature.parameters.values())
    for index, parameter in enumerate(parameters):
        if type(parameter.default).__name__ == "_NotProvided":
            parameters[index] = parameter.replace(default="<not-provided>")
        elif dynamic_device and parameter.name == "device":
            parameters[index] = parameter.replace(default=AUTO_DEVICE_DEFAULT)
    signature = signature.replace(parameters=parameters)
    return str(signature)


def public_identity(value) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def cli_actions() -> list[dict[str, object]]:
    actions = []
    for action in get_parser()._actions:
        default = action.default
        if action.dest == "device":
            default = AUTO_DEVICE_DEFAULT
        actions.append(
            {
                "dest": action.dest,
                "option_strings": list(action.option_strings),
                "nargs": action.nargs,
                "required": action.required,
                "default": json_value(default),
                "choices": list(action.choices) if action.choices is not None else None,
                "type": getattr(action.type, "__name__", None),
            }
        )
    return actions


def cli_behavior() -> dict[str, object]:
    stderr = io.StringIO()
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli_main([])
    except SystemExit as error:
        exit_code = error.code
    else:
        exit_code = 0
    return {
        "program": get_parser().prog,
        "no_tracks_exit": exit_code,
        "no_tracks_stdout": stdout.getvalue(),
        "no_tracks_stderr": stderr.getvalue(),
        "actions": cli_actions(),
    }


def build_snapshot() -> dict[str, object]:
    public_names = (
        "DemucsSeparator",
        "DemucsSession",
        "separate",
        "separate_file",
        "separate_tensor",
        "CHECKPOINT_CATALOG",
        "checkpoint_catalog",
        "checkpoint_config_path",
        "get_checkpoint_metadata",
        "validate_checkpoint_config",
    )
    signatures = {
        "pretrained.get_model": normalized_signature(get_model),
        "api.Separator": normalized_signature(Separator, dynamic_device=True),
        "api.Separator.update_parameter": normalized_signature(Separator.update_parameter),
        "api.Separator.separate_tensor": normalized_signature(Separator.separate_tensor),
        "api.Separator.separate_audio_file": normalized_signature(Separator.separate_audio_file),
        "clean_api.DemucsSeparator": normalized_signature(clean_api.DemucsSeparator),
        "clean_api.DemucsSeparator.load": normalized_signature(clean_api.DemucsSeparator.load),
        "clean_api.DemucsSeparator.infer": normalized_signature(clean_api.DemucsSeparator.infer),
        "clean_api.DemucsSeparator.cache_info": normalized_signature(clean_api.DemucsSeparator.cache_info),
        "clean_api.DemucsSeparator.__call__": normalized_signature(clean_api.DemucsSeparator.__call__),
        "clean_api.separate": normalized_signature(clean_api.separate),
        "clean_api.separate_file": normalized_signature(clean_api.separate_file),
        "clean_api.separate_tensor": normalized_signature(clean_api.separate_tensor),
        "checkpoint_catalog.get_checkpoint_metadata": normalized_signature(get_checkpoint_metadata),
        "states.load_model": normalized_signature(load_model),
    }
    catalog = {key: dict(value) for key, value in CHECKPOINT_CATALOG.items()}
    return {
        "schema_version": 1,
        "package_exports": {
            name: public_identity(getattr(demucs_infer, name)) if callable(getattr(demucs_infer, name)) else type(getattr(demucs_infer, name)).__name__
            for name in public_names
        },
        "clean_api_all": list(clean_api.__all__),
        "demucs_session_is_alias": clean_api.DemucsSession is clean_api.DemucsSeparator,
        "signatures": signatures,
        "checkpoint_catalog": catalog,
        "htdemucs_metadata": get_checkpoint_metadata("htdemucs"),
        "cli": cli_behavior(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if not args.out and not args.compare:
        parser.error("one of --out or --compare is required")
    snapshot = build_snapshot()
    if args.compare:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        if snapshot != expected:
            raise SystemExit("public contract differs from the Phase 0 snapshot")
        print(f"public contract matches {args.compare}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote public contract snapshot to {args.out}")


if __name__ == "__main__":
    main()
