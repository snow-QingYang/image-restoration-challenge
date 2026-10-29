"""The exact shared-weight NAFNet used by the three-stage training route.

Stages ``scratch`` and ``native`` use WideSharedNAFNet directly. Stage ``d4``
wraps that same network in TrainableSelfEnsemble: eight fixed spatial views,
one shared set of weights, and an unclipped average in original coordinates.
The default plain network has 286,563 parameters; D4 adds no parameters.

This file contains the complete model implementation. Class member names
and forward arithmetic match the accepted three-stage experiment so its
plain and model.* EMA state dicts remain compatible. Inputs are RGB NCHW
tensors, with H/W divisible by four.
D4 additionally requires square inputs. Training outputs are not clipped.

NAF equations are adapted from "Simple Baselines for Image Restoration":
https://arxiv.org/abs/2204.04676
https://github.com/megvii-research/NAFNet
The within-stage convolution sharing is this project's custom architecture.

Copyright (c) 2022 megvii-model

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections.abc import Mapping

import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each spatial position."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        left, right = x.chunk(2, dim=1)
        return left * right


class SharedNAFConvolutions(nn.Module):
    """One stage's shared convolutions; norms and residual gates stay separate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 2 * channels, 1)
        self.conv2 = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1,
                               groups=2 * channels)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.conv4 = nn.Conv2d(channels, 2 * channels, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)

    def spatial(self, x):
        features = self.gate(self.conv2(self.conv1(x)))
        return self.conv3(features * self.sca(features))

    def channel(self, x):
        return self.conv5(self.gate(self.conv4(x)))


class NAFStepState(nn.Module):
    """Independent normalization and residual gates for one unfolded step."""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))


class SharedNAFStage(nn.Module):
    """Reuse one convolution cell, with independent norms/gates at each depth."""

    def __init__(self, channels, depth):
        super().__init__()
        if type(channels) is not int or channels < 1 or type(depth) is not int or depth < 1:
            raise ValueError("channels and depth must be positive integers")
        self.shared = SharedNAFConvolutions(channels)
        self.steps = nn.ModuleList(NAFStepState(channels) for _ in range(depth))

    def forward(self, x):
        for step in self.steps:
            x = x + self.shared.spatial(step.norm1(x)) * step.beta
            x = x + self.shared.channel(step.norm2(x)) * step.gamma
        return x


class WideSharedNAFNet(nn.Module):
    """Residual U-Net with one shared convolution cell per resolution/stage.

    Default widths are 32/64/128; unfolded depths are 2/2/7/2/2. Encoder
    and decoder cells are distinct even where their channel widths match.
    Zero-initializing the final RGB projection makes scratch start as the
    identity mapping. Convolution initialization order matches the source.
    """

    def __init__(self, width=32, enc_blk_nums=(2, 2), middle_blk_num=7,
                 dec_blk_nums=(2, 2)):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if len(enc_blk_nums) != 2 or len(dec_blk_nums) != 2:
            raise ValueError("WideSharedNAFNet requires two encoder and decoder stages")
        if any(type(count) is not int or count < 1
               for count in (*enc_blk_nums, middle_blk_num, *dec_blk_nums)):
            raise ValueError("all stage depths must be positive integers")

        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = width
        for depth in enc_blk_nums:
            self.encoders.append(SharedNAFStage(channels, depth))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle_blks = SharedNAFStage(channels, middle_blk_num)
        for depth in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, 2 * channels, 1, bias=False),
                                          nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(SharedNAFStage(channels, depth))

    def forward(self, x):
        # Extract features, save encoder skips, then restore resolution.
        features = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle_blks(features)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            features = decoder(up(features) + skip)
        # Learn the restoration residual; the training output stays unclipped.
        return x + self.ending(features)


class SelfEnsemble(nn.Module):
    """Historical clipped inference wrapper, retained as the D4 base class."""

    def __init__(self, model, mode="d4"):
        super().__init__()
        if mode not in ("d4", "flips4"):
            raise ValueError("mode must be 'd4' or 'flips4'")
        self.model = model
        self.mode = mode
        self.transforms = tuple((transpose, flip_h, flip_v)
                                for transpose in ((False, True) if mode == "d4" else (False,))
                                for flip_h in (False, True)
                                for flip_v in (False, True))

    def forward(self, x):
        total = None
        for transpose, flip_h, flip_v in self.transforms:
            transformed = x.transpose(-2, -1) if transpose else x
            axes = tuple(axis for enabled, axis in ((flip_h, -1), (flip_v, -2)) if enabled)
            if axes:
                transformed = torch.flip(transformed, axes)
            prediction = self.model(transformed)
            if axes:
                prediction = torch.flip(prediction, axes)
            if transpose:
                prediction = prediction.transpose(-2, -1)
            total = prediction if total is None else total + prediction
        return (total / len(self.transforms)).clamp(0, 1)


class TrainableSelfEnsemble(SelfEnsemble):
    """Eight spatial views in one batch, with an unclipped shared-weight mean.

    Backpropagation through the average updates the single base network.
    The wrapper adds no parameters; its state_dict uses the prefix model.*.
    View order, inverse transforms, and left-to-right sum match the source.
    """

    def __init__(self, model, mode="d4"):
        super().__init__(model, mode)
        stochastic = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                      nn.AlphaDropout, nn.FeatureAlphaDropout)
        for name, module in model.named_modules():
            if isinstance(module, stochastic) or getattr(module, "track_running_stats", False):
                raise ValueError(f"Batched self-ensemble requires an independent deterministic base; unsupported module {name!r}")

    def forward(self, x):
        if not torch.jit.is_tracing() and (x.ndim != 4 or x.shape[-2] != x.shape[-1]):
            raise ValueError("Batched self-ensemble expects square NCHW inputs")
        views = []
        for transpose, flip_h, flip_v in self.transforms:
            transformed = x.transpose(-2, -1) if transpose else x
            axes = tuple(axis for enabled, axis in ((flip_h, -1), (flip_v, -2)) if enabled)
            views.append(torch.flip(transformed, axes) if axes else transformed)
        predictions = self.model(torch.cat(views, dim=0)).chunk(len(self.transforms), dim=0)
        total = None
        for prediction, (transpose, flip_h, flip_v) in zip(predictions, self.transforms):
            axes = tuple(axis for enabled, axis in ((flip_h, -1), (flip_v, -2)) if enabled)
            if axes:
                prediction = torch.flip(prediction, axes)
            if transpose:
                prediction = prediction.transpose(-2, -1)
            total = prediction if total is None else total + prediction
        return total / len(self.transforms)

    def load_pretrained_state_dict(self, state_dict):
        """Validate and load plain EMA tensors into the single base network."""
        _validate_state_dict(self.model, state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        return {"method": "wrap_pretrained_model_state_dict", "state_dict_prefix": "model.",
                "loaded_base_keys": len(state_dict), "new_parameters": 0,
                "base_model_class": type(self.model).__name__, "self_ensemble_mode": self.mode,
                "transform_count": len(self.transforms), "forward": "unclipped_batched_mean",
                "output_clipping": "PSNR metric and ONNX export wrapper only"}


def _validate_state_dict(model, state_dict):
    """Check every key/shape/dtype before changing any model parameters."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be the EMA tensor mapping, not a checkpoint path")
    if any(not isinstance(name, str) for name in state_dict):
        raise TypeError("all EMA state_dict keys must be strings")
    expected = model.state_dict()
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    incompatible = [name for name in expected.keys() & state_dict.keys()
                    if not isinstance(state_dict[name], torch.Tensor)
                    or state_dict[name].shape != expected[name].shape
                    or state_dict[name].dtype != expected[name].dtype]
    if missing or unexpected or incompatible:
        raise ValueError("EMA tensors must exactly match model keys, shapes and dtypes; "
                         f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
                         f"incompatible={sorted(incompatible)}")


def build_model(config):
    """Build the configured model, following the assignment's starter API.

    ``config`` is the dictionary read from config.json. ``stage`` selects
    ``scratch``/``native`` (one plain network) or ``d4`` (the same network
    with eight-view averaging). If stage is omitted, ``architecture``
    selects ``shared_nafnet`` or ``shared_nafnet_d4``; the latter is the
    default final submission model. Passing a stage string is also accepted
    for older checkpoints and scripts.

    Architecture dimensions come from width/enc_blk_nums/middle_blk_num/
    dec_blk_nums. Their defaults reproduce the accepted 286,563-parameter
    network exactly. The D4 wrapper introduces no additional parameters.
    """
    if isinstance(config, str):
        config = {"stage": config}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a configuration mapping or a stage string")
    stage = config.get("stage")
    if stage is None:
        architecture = config.get("architecture", "shared_nafnet_d4")
        architectures = {
            "shared_nafnet": "native",
            "nafnet_wide_shared": "native",
            "shared_nafnet_d4": "d4",
            "trainable_d4_of_nafnet_wide_shared": "d4",
        }
        if architecture not in architectures:
            raise ValueError(f"unsupported architecture {architecture!r}")
        stage = architectures[architecture]
    if stage not in ("scratch", "native", "d4"):
        raise ValueError("stage must be 'scratch', 'native', or 'd4'")
    base = WideSharedNAFNet(width=config.get("width", 32),
                            enc_blk_nums=config.get("enc_blk_nums", (2, 2)),
                            middle_blk_num=config.get("middle_blk_num", 7),
                            dec_blk_nums=config.get("dec_blk_nums", (2, 2)))
    return TrainableSelfEnsemble(base, mode="d4") if stage == "d4" else base


def load_ema_weights(model, state_dict):
    """Load EMA tensors strictly, including the plain-native to D4 transition.

    The caller extracts ``checkpoint['model']`` (the EMA tensor mapping in
    both the historical and new checkpoint formats) and supplies it here.
    A D4 destination accepts either its own complete ``model.*`` mapping or
    the plain base network's complete mapping. Other mappings are rejected;
    no key is silently removed and no shape or dtype is silently converted.
    This loads model tensors only: stage transitions create a fresh optimizer
    and scheduler rather than resuming their previous states.
    """
    if isinstance(model, TrainableSelfEnsemble):
        if isinstance(state_dict, Mapping) and not any(
                isinstance(name, str) and name.startswith("model.") for name in state_dict):
            return model.load_pretrained_state_dict(state_dict)
    _validate_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    return {"method": "strict_same_architecture_ema", "loaded_keys": len(state_dict),
            "model_class": type(model).__name__}
