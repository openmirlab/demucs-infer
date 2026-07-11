#!/usr/bin/env python3
"""ADOPT P1 -- wiener vendoring regression fixture generator.

htdemucs's shipped config never calls ``demucs_infer.wiener.wiener`` (it
always runs with ``cac=True``, which short-circuits ``_mask()`` before the
wiener path -- see hdemucs.py/htdemucs.py's ``_mask``/``_wiener``). So
tests/test_wiener_vendored.py cannot piggyback on the htdemucs baseline
fixture (tests/fixtures/baseline_htdemucs.json) to catch a regression in the
vendored code.

This tool was used once, with openunmix temporarily reinstalled
(``uv pip install openunmix==1.3.0``, undone afterwards -- openunmix is not
a dependency of this package), to record reference outputs from
``openunmix.filtering.wiener`` across the softmask/residual/iterations
combinations that htdemucs's own config never exercises. It wrote
tests/fixtures/wiener_reference.json, which tests/test_wiener_vendored.py
now checks the *vendored* demucs_infer.wiener.wiener against on every run
(no openunmix import needed at test time).

Verified bit-identical to demucs_infer.wiener at capture time via
torch.equal(); see the ADOPT campaign report for the one-off real
end-to-end check (mdx_extra, whose cac=False bag members do call this at
inference) that additionally confirmed this in a full forward pass.

Reads: openunmix.filtering (only at fixture-capture time, not a runtime dep)
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

CASES = [
    {"softmask": sm, "residual": res, "iterations": it}
    for sm in (False, True)
    for res in (False, True)
    for it in (0, 1, 2)
]


def digest(t: torch.Tensor) -> dict:
    arr = t.detach().cpu().contiguous().to(torch.float64).numpy()
    return {
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                         default=REPO_ROOT / "tests" / "fixtures" / "wiener_reference.json")
    parser.add_argument("--use-openunmix", action="store_true",
                         help="Use openunmix.filtering.wiener as the source of truth "
                              "(requires temporarily: uv pip install openunmix==1.3.0). "
                              "Default uses the vendored copy (for regenerating after an "
                              "intentional, verified change).")
    args = parser.parse_args()

    if args.use_openunmix:
        from openunmix.filtering import wiener
    else:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from demucs_infer.wiener import wiener

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    nb_frames, nb_bins, nb_channels, nb_sources = 5, 17, 2, 4

    fixture = {"meta": {"seed": 0, "shape": [nb_frames, nb_bins, nb_channels, nb_sources],
                         "source": "openunmix" if args.use_openunmix else "vendored"},
               "inputs": {}, "cases": []}

    targets = torch.rand(nb_frames, nb_bins, nb_channels, nb_sources, generator=gen, dtype=torch.float64)
    mix_stft = torch.randn(nb_frames, nb_bins, nb_channels, 2, generator=gen, dtype=torch.float64)
    fixture["inputs"]["targets_sha256"] = hashlib.sha256(targets.numpy().tobytes()).hexdigest()
    fixture["inputs"]["mix_stft_sha256"] = hashlib.sha256(mix_stft.numpy().tobytes()).hexdigest()

    for case in CASES:
        out = wiener(targets.clone(), mix_stft.clone(), **case)
        fixture["cases"].append({**case, "output": digest(out)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
