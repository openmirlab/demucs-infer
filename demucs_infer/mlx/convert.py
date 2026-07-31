"""PyTorch -> MLX weight conversion for `HDemucsMLX`/`HTDemucsMLX`, plus a
strict load gate.

`convert_state_dict` walks the *live* Torch model's `named_modules()` to
learn each parameter's true layout (`Conv1d` vs `Conv2d` vs their transposed
counterparts) rather than guessing from the parameter name -- the same
approach upstream's `mlx_convert.py` uses, and more robust than a purely
regex-driven remap (see the sibling `bs-roformer-infer`/`mdxnet-infer`
packages' `mlx/convert.py`, which have no live Torch module tree to consult
because their seam is built from a checkpoint path, not an already-loaded
model). Handles: Conv1d/Conv2d/ConvTranspose1d/ConvTranspose2d layout
transposition, GroupNorm parameter passthrough (this port's `GroupNormNCL`/
`GroupNormNCHW` use the same flat `weight`/`bias` names Torch's `GroupNorm`
does, so no renaming is needed there), Torch's packed bidirectional
`nn.LSTM` split into this port's separate `forward_lstms`/`backward_lstms`
lists, and Torch's packed `nn.MultiheadAttention.in_proj_weight`/
`in_proj_bias` split into MLX's `query_proj`/`key_proj`/`value_proj`.

`load_converted_weights` then diffs the model's own parameter keys (via
`mlx.utils.tree_flatten`) against the converted weight keys and raises a
`ValueError` naming the mismatch *before* calling `load_weights` -- plain
`model.load_weights(..., strict=False)` silently discards any key that
doesn't match the module tree, which would leave a whole layer at random
initialization and produce confident garbage with no error. This is the
guard named in the non-negotiables: it has already caught real bugs in three
prior ports of this same pattern (both sibling packages, plus one bug in the
upstream reference itself), so it is applied here unconditionally rather
than re-derived per port.

Adapted from `mlx-audio-separator`'s `demucs_mlx/mlx_convert.py`
(`convert_state_dict`, `convert_conv_weight`) -- see `hdemucs.py`'s module
docstring for the full vendoring/attribution note; this file's
`load_converted_weights` is new, not upstream (upstream's own loader calls
`load_weights(..., strict=False)` directly, which is exactly the failure
mode this function exists to prevent).

Reads: mlx.core, mlx.utils (tree_flatten), numpy, torch (typing-only, for
isinstance checks against the live Torch module tree)
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def _to_numpy(value) -> np.ndarray:
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.array(value)


def _conv_layout(module_type: str) -> tp.Optional[str]:
    return {
        "Conv1d": "conv1d",
        "ConvTranspose1d": "conv_transpose1d",
        "Conv2d": "conv2d",
        "ConvTranspose2d": "conv_transpose2d",
    }.get(module_type)


def _transpose_conv_weight(weight: np.ndarray, layout: str) -> np.ndarray:
    if layout == "conv1d":
        return np.transpose(weight, (0, 2, 1))
    if layout == "conv_transpose1d":
        return np.transpose(weight, (1, 2, 0))
    if layout == "conv2d":
        return np.transpose(weight, (0, 2, 3, 1))
    if layout == "conv_transpose2d":
        return np.transpose(weight, (1, 2, 3, 0))
    raise ValueError(f"unknown conv layout: {layout}")


def convert_state_dict(torch_model) -> tp.Dict[str, mx.array]:
    """Convert `torch_model.state_dict()` to a flat MLX-weight-name dict.

    `torch_model` must be the live Torch `HDemucs`/`HTDemucs` instance (not
    just its `state_dict()`), so this can walk `named_modules()` to learn
    each Conv layer's true layout.

    Every Conv1d/Conv2d/ConvTranspose1d/ConvTranspose2d in this port is
    wrapped in a `layers.Conv1dNCL`/`Conv2dNCHW`/... class for its NCL/NCHW
    <-> MLX-native NLC/NHWC transpose (see `layers.py`'s module docstring),
    so every such tensor lives one level deeper than Torch's flat naming
    (`encoder.0.conv.weight` -> `encoder.0.conv.conv.weight`,
    `decoder.0.conv_tr.weight` -> `decoder.0.conv_tr.conv.weight`, and so
    on) -- `.conv` is inserted before the final `weight`/`bias` segment for
    every key `named_modules()` identified as one of those four Torch
    classes. Caught by `load_converted_weights` raising on the very first
    conversion attempt against a real checkpoint (initially only reproduced
    for the four `channel_up/downsampler[_t]` names before the same pattern
    turned out to be universal -- see this function's own history in
    `CHANGELOG.md`).
    """
    module_layout: tp.Dict[str, str] = {}
    for module_name, module in torch_model.named_modules():
        layout = _conv_layout(type(module).__name__)
        if layout is not None:
            module_layout[f"{module_name}.weight"] = layout

    flat: tp.Dict[str, mx.array] = {}
    for name, param in torch_model.state_dict().items():
        np_param = _to_numpy(param)
        layout = module_layout.get(name)
        if layout is not None:
            np_param = _transpose_conv_weight(np_param, layout)
        flat[name] = mx.array(np_param)

    _remap_bidirectional_lstm(flat)
    _remap_multihead_attention(flat)
    _remap_wrapped_conv_layers(flat, module_layout)
    return flat


def _remap_wrapped_conv_layers(flat: tp.Dict[str, mx.array], module_layout: tp.Dict[str, str]) -> None:
    conv_bases = {name[: -len(".weight")] for name in module_layout}
    for base in conv_bases:
        for suffix in (".weight", ".bias"):
            name = base + suffix
            if name in flat:
                flat[f"{base}.conv{suffix}"] = flat.pop(name)


def _remap_bidirectional_lstm(flat: tp.Dict[str, mx.array]) -> None:
    """Torch's single bidirectional `nn.LSTM` (`weight_ih_l{N}[_reverse]`,
    `bias_ih_l{N}[_reverse]`, `bias_hh_l{N}[_reverse]`) -> this port's
    `forward_lstms`/`backward_lstms` lists (`Wx`, `Wh`, `bias` -- MLX's
    `nn.LSTM` sums the two Torch biases into one, since it only has one bias
    term per gate)."""
    bias_parts: tp.Dict[tp.Tuple[str, str, int], tp.Dict[str, mx.array]] = {}
    for name in list(flat.keys()):
        if ".lstm." not in name:
            continue
        prefix, rest = name.split(".lstm.", 1)
        is_reverse = rest.endswith("_reverse")
        if is_reverse:
            rest = rest[: -len("_reverse")]
        if "_l" not in rest:
            continue
        base, layer_str = rest.rsplit("_l", 1)
        if not layer_str.isdigit():
            continue
        layer = int(layer_str)
        direction = "backward_lstms" if is_reverse else "forward_lstms"
        if base == "weight_ih":
            flat[f"{prefix}.{direction}.{layer}.Wx"] = flat.pop(name)
        elif base == "weight_hh":
            flat[f"{prefix}.{direction}.{layer}.Wh"] = flat.pop(name)
        elif base in ("bias_ih", "bias_hh"):
            key = (prefix, direction, layer)
            bias_parts.setdefault(key, {})[base] = flat.pop(name)

    for (prefix, direction, layer), parts in bias_parts.items():
        bias_ih = parts.get("bias_ih")
        bias_hh = parts.get("bias_hh")
        if bias_ih is not None and bias_hh is not None:
            bias = bias_ih + bias_hh
        else:
            bias = bias_ih if bias_ih is not None else bias_hh
        flat[f"{prefix}.{direction}.{layer}.bias"] = bias


def _remap_multihead_attention(flat: tp.Dict[str, mx.array]) -> None:
    """Torch's `nn.MultiheadAttention` (`{self,cross}_attn.in_proj_weight`,
    `in_proj_bias`, `out_proj.{weight,bias}`) -> MLX's `nn.MultiHeadAttention`
    (`attn.{query,key,value}_proj.{weight,bias}`, `attn.out_proj.*`)."""
    for name in list(flat.keys()):
        # `self_attn` (Torch's `nn.TransformerEncoderLayer`-inherited name)
        # -> `attn` (this port's `TransformerEncoderLayer.attn`). `cross_attn`
        # needs no rename: both Torch's `CrossTransformerEncoderLayer` and
        # this port's use that same name -- renaming it too was a real bug
        # caught by `load_converted_weights` (every cross layer's attention
        # ended up dropped, and an unrelated classic layer's `attn.*` looked
        # unmatched, because both got the same `.attn.` name after rename).
        if ".self_attn." in name:
            new_name = name.replace(".self_attn.", ".attn.")
        elif ".cross_attn." in name:
            new_name = name
        else:
            continue

        if name.endswith(".in_proj_weight"):
            weight = np.array(flat.pop(name))
            dim = weight.shape[0] // 3
            base = new_name[: -len(".in_proj_weight")]
            flat[f"{base}.query_proj.weight"] = mx.array(weight[:dim])
            flat[f"{base}.key_proj.weight"] = mx.array(weight[dim:2 * dim])
            flat[f"{base}.value_proj.weight"] = mx.array(weight[2 * dim:])
        elif name.endswith(".in_proj_bias"):
            bias = np.array(flat.pop(name))
            dim = bias.shape[0] // 3
            base = new_name[: -len(".in_proj_bias")]
            flat[f"{base}.query_proj.bias"] = mx.array(bias[:dim])
            flat[f"{base}.key_proj.bias"] = mx.array(bias[dim:2 * dim])
            flat[f"{base}.value_proj.bias"] = mx.array(bias[2 * dim:])
        elif ".out_proj." in name:
            if new_name != name:
                flat[new_name] = flat.pop(name)
        else:
            # Unexpected suffix under self_attn/cross_attn (shouldn't happen
            # for the dense-attention path this port supports) -- leave the
            # renamed key so a mismatch surfaces via load_converted_weights
            # rather than silently vanishing.
            if new_name != name:
                flat[new_name] = flat.pop(name)


def load_converted_weights(model, mlx_weights: tp.Dict[str, mx.array]) -> None:
    """Load `mlx_weights` into `model`, refusing a silent partial load.

    `model.load_weights(..., strict=False)` on its own accepts any degree of
    mismatch between the checkpoint and the module tree, dropping whatever
    doesn't line up without a warning. This checks first: every one of the
    model's own parameter keys (from `mlx.utils.tree_flatten(model.parameters())`)
    must be present in `mlx_weights`, and every key in `mlx_weights` must be
    consumed by the model -- otherwise a `ValueError` is raised naming counts
    and up to 5 example keys on each side, so a conversion bug or a mismatched
    checkpoint fails loudly instead of loading a partially-random model.
    """
    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    weight_keys = set(mlx_weights.keys())

    unmatched_model = sorted(model_keys - weight_keys)
    dropped_weights = sorted(weight_keys - model_keys)

    if unmatched_model or dropped_weights:
        parts = []
        if unmatched_model:
            example = ", ".join(unmatched_model[:5])
            parts.append(f"{len(unmatched_model)} model parameters unmatched (e.g. {example})")
        if dropped_weights:
            example = ", ".join(dropped_weights[:5])
            parts.append(f"{len(dropped_weights)} converted tensors dropped (e.g. {example})")
        raise ValueError("MLX weight conversion incomplete: " + ", ".join(parts))

    model.load_weights(list(mlx_weights.items()), strict=False)
