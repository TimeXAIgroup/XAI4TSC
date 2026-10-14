"""Unit tests for model persistence/registry — light torch use, no training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc.models.base import ModelBase, predict_probs
from xai4tsc.models.models import (
    FCN,
    LSTM,
    MODELS,
    XLSTM,
    LeNet,
    PatchTST,
    ResNet,
    _parallel_mlstm,
    _ResidualBlock,
    load_ext_model,
    load_model,
    register_model,
    save_model,
)

# Tiny FCN config so construction is instant.
_TINY_FCN = {
    "model": "FCN",
    "init_params": {
        "in_channels": 1,
        "num_classes": 2,
        "filters": [4, 8, 4],
        "kernel_sizes": [7, 5, 3],
    },
}


class _TinyModel(ModelBase):
    """Minimal concrete ModelBase used for registry/save tests."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x.mean(-1))


# ── register_model ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_register_model_adds_to_registry_and_instantiates(tmp_path):
    register_model("tinymodel", _TinyModel)
    try:
        assert MODELS["tinymodel"] is _TinyModel
        model = load_model(
            {"model": "tinymodel", "init_params": {"in_channels": 1, "num_classes": 2}},
            device="cpu",
            save_path=tmp_path,
        )
        assert isinstance(model, _TinyModel)
        assert model.name == "tinymodel"
    finally:
        MODELS.pop("tinymodel", None)


@pytest.mark.unit
def test_register_model_rejects_non_modelbase():
    from torch import nn

    class _PlainModule(nn.Module):
        pass

    with pytest.raises(TypeError, match="ModelBase subclass"):
        register_model("plainmodule", _PlainModule)
    assert "plainmodule" not in MODELS


# ── save_model ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_save_model_writes_state_and_config(tmp_path):
    model = _TinyModel()
    save_model(model, ckpt={"epoch": 1}, save_path=tmp_path)
    assert (tmp_path / "model_dict.pt").exists()
    assert (tmp_path / "model_dict_config.json").exists()


@pytest.mark.unit
def test_save_model_writes_optimizer_when_supplied(tmp_path):
    model = _TinyModel()
    optim = torch.optim.Adam(model.parameters())
    save_model(model, ckpt={}, optim_state_dict=optim.state_dict(), save_path=tmp_path)
    assert (tmp_path / "optim_dict.pt").exists()


@pytest.mark.unit
def test_save_model_without_path_raises():
    with pytest.raises(ValueError):
        save_model(_TinyModel(), ckpt={})


# ── load_model (registry path) ────────────────────────────────────────────────


@pytest.mark.unit
def test_load_model_registry_sets_operational_attrs(tmp_path):
    model = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)
    assert model.name == "FCN"
    assert model.device == torch.device("cpu")
    assert model.save_path == tmp_path
    # init_params are filtered against the constructor signature and stored.
    assert model._init_params["in_channels"] == 1
    assert model._init_params["num_classes"] == 2


_EXT_MODEL_SOURCE = """
from torch import nn

from xai4tsc.models.base import ModelBase


class ExtModel(ModelBase):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x.mean(-1))
"""


@pytest.mark.unit
def test_external_class_checkpoint_keeps_its_init_params(tmp_path):
    # The external branch used to overwrite _init_params with {} right after
    # load_ext_model had recorded them, which left the checkpoint sidecar
    # unable to say how to rebuild the model.
    source = tmp_path / "ext_model.py"
    source.write_text(_EXT_MODEL_SOURCE)
    config = {
        "model": "ExtModelName",
        "class_name": "ExtModel",
        "class_path": str(source),
        "init_params": {"in_channels": 1, "num_classes": 3},
    }

    built = load_ext_model(
        model_name="ExtModelName",
        class_name="ExtModel",
        model_path=source,
        model_params=config["init_params"],
    )
    ckpt = tmp_path / "ExtModelName_epoch_1.pt"
    torch.save(built.state_dict(), ckpt)

    reloaded = load_model(
        {**config, "model_ckpt_path": str(ckpt)}, device="cpu", save_path=tmp_path
    )

    assert reloaded._init_params == {"in_channels": 1, "num_classes": 3}
    # With the arguments recorded, the sidecar can now be written and reloaded.
    reloaded.save_model(tmp_path / "ExtModelName_epoch_2.pt", epoch=2)
    cfg = json.loads((tmp_path / "ExtModelName_epoch_2_config.json").read_text())
    assert cfg["init_params"] == {"in_channels": 1, "num_classes": 3}


@pytest.mark.unit
def test_direct_construction_names_the_model_after_its_class():
    # A model built with its constructor never reaches load_model, so it used to
    # keep name=None and write its files as "None_epoch_2.pt".  The class name is
    # always available and is what load_model would have set anyway.
    model = FCN(in_channels=1, num_classes=2)
    assert model.name == "FCN"
    assert _TinyModel().name == "_TinyModel"
    # Only the name is defaulted: a destination is still the caller's to supply.
    assert model.save_path is None


@pytest.mark.unit
def test_load_model_unknown_name_raises():
    with pytest.raises(ValueError):
        load_model({"model": "no_such_model"})


# ── save → load round trip ────────────────────────────────────────────────────


@pytest.mark.unit
def test_save_then_load_round_trip_preserves_weights(tmp_path):
    original = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)
    # The config file (= ckpt) must carry the init_params so the reloaded model is
    # built with a matching architecture before the state dict is applied.
    save_model(original, ckpt=_TINY_FCN, save_path=tmp_path)

    reloaded = load_model(
        {**_TINY_FCN, "model_ckpt_path": str(tmp_path / "model_dict.pt")},
        device="cpu",
        save_path=tmp_path,
    )

    orig_state = original.state_dict()
    new_state = reloaded.state_dict()
    assert orig_state.keys() == new_state.keys()
    for key, tensor in orig_state.items():
        assert torch.equal(tensor, new_state[key])


# ── FCN construction ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_fcn_init_constructs_blocks():
    model = load_model(
        {
            "model": "FCN",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "filters": (4, 8, 4),
                "kernel_sizes": (3, 3, 3),
            },
        }
    )
    n_conv = sum(1 for m in model.features if isinstance(m, nn.Conv1d))
    assert n_conv == 3
    assert model.classifier.out_features == 2


# ── LeNet (Wagner 2024) / ResNet (Wang 2017) ─────────────────────────────────────

# Tiny configs so construction/forward are instant.
_TINY_LENET = {
    "model": "LeNet",
    "init_params": {"in_channels": 1, "num_classes": 2, "head_hidden": 8},
}
_TINY_RESNET = {
    "model": "ResNet",
    "init_params": {"in_channels": 1, "num_classes": 2, "n_feature_maps": 4},
}


@pytest.mark.unit
def test_lenet_forward_output_shape():
    model = LeNet(in_channels=1, num_classes=3, head_hidden=8)
    model.eval()
    out = model(torch.zeros(2, 1, 128))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_lenet_multivariate_forward():
    model = LeNet(in_channels=7, num_classes=4, head_hidden=8)
    model.eval()
    out = model(torch.zeros(2, 7, 128))
    assert out.shape == (2, 4)


@pytest.mark.unit
@pytest.mark.parametrize("length", [40, 256])
def test_lenet_is_length_agnostic(length):
    # The global AdaptiveAvgPool makes the FC head independent of T.
    model = LeNet(in_channels=1, num_classes=3, head_hidden=8)
    model.eval()
    out = model(torch.zeros(2, 1, length))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_resnet_default_forward_output_shape():
    model = ResNet(in_channels=1, num_classes=3, n_feature_maps=4)
    model.eval()
    out = model(torch.zeros(2, 1, 128))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_resnet_multivariate_forward():
    model = ResNet(in_channels=12, num_classes=5, n_feature_maps=4)
    model.eval()
    out = model(torch.zeros(2, 12, 256))
    assert out.shape == (2, 5)


@pytest.mark.unit
@pytest.mark.parametrize("length", [40, 256])
def test_resnet_is_length_agnostic(length):
    # padding="same" keeps T intact through the blocks; GAP collapses it.
    model = ResNet(in_channels=1, num_classes=3, n_feature_maps=4)
    model.eval()
    out = model(torch.zeros(2, 1, length))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_resnet_custom_hyperparams_via_load_model():
    # The configurable params survive the YAML-style init_params -> dict_to_args path.
    model = load_model(
        {
            "model": "ResNet",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "n_feature_maps": 8,
                "kernel_sizes": [5, 3, 3],
            },
        },
        device="cpu",
    )
    out = model(torch.zeros(2, 1, 128))
    assert out.shape == (2, 2)


@pytest.mark.unit
def test_resnet_matches_wang2017_architecture():
    # Structural correctness vs the Wang et al. (2017) ResNet spec: 3 residual
    # blocks with channel widths F, 2F, 2F; each block has 3 convs (kernels
    # 8/5/3); shortcut is a 1x1 conv+BN when the width changes else a plain BN;
    # the block ReLU is applied AFTER the residual sum (so conv_block ends on BN).
    model = ResNet(in_channels=3, num_classes=5)  # default n_feature_maps=64
    assert len(model.blocks) == 3
    for blk, (w_in, w_out) in zip(
        model.blocks, [(3, 64), (64, 128), (128, 128)], strict=True
    ):
        assert isinstance(blk, _ResidualBlock)
        convs = [m for m in blk.conv_block if isinstance(m, nn.Conv1d)]
        assert [c.kernel_size[0] for c in convs] == [8, 5, 3]
        assert convs[0].in_channels == w_in
        assert all(c.out_channels == w_out for c in convs)
        assert isinstance(blk.conv_block[-1], nn.BatchNorm1d)  # ReLU after the sum
        assert isinstance(blk.act, nn.ReLU)
        if w_in != w_out:
            assert isinstance(blk.shortcut, nn.Sequential)
            assert blk.shortcut[0].kernel_size[0] == 1
        else:
            assert isinstance(blk.shortcut, nn.BatchNorm1d)
    assert model.classifier.in_features == 128
    assert model.classifier.out_features == 5


# ── LSTM ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_lstm_forward_output_shape():
    model = LSTM(in_channels=1, num_classes=3, hidden_size=8, num_layers=1)
    model.eval()
    out = model(torch.zeros(2, 1, 64))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_lstm_multivariate_forward():
    model = LSTM(in_channels=7, num_classes=4, hidden_size=8, num_layers=1)
    model.eval()
    out = model(torch.zeros(2, 7, 64))
    assert out.shape == (2, 4)


@pytest.mark.unit
@pytest.mark.parametrize("length", [16, 200])
def test_lstm_is_length_agnostic(length):
    # Mean-pooling over time makes the head independent of T.
    model = LSTM(in_channels=1, num_classes=3, hidden_size=8, num_layers=1)
    model.eval()
    out = model(torch.zeros(2, 1, length))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_lstm_unidirectional_head_width():
    # bidirectional=False halves the classifier input width vs the default.
    model = LSTM(
        in_channels=1, num_classes=3, hidden_size=8, num_layers=1, bidirectional=False
    )
    assert model.classifier.in_features == 8
    bi = LSTM(in_channels=1, num_classes=3, hidden_size=8, num_layers=1)
    assert bi.classifier.in_features == 16


@pytest.mark.unit
def test_lstm_via_load_model():
    # Hyperparameters survive the YAML-style init_params -> dict_to_args path.
    model = load_model(
        {
            "model": "LSTM",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "hidden_size": 8,
                "num_layers": 1,
                "bidirectional": False,
            },
        },
        device="cpu",
    )
    out = model(torch.zeros(2, 1, 64))
    assert out.shape == (2, 2)


# ── PatchTST ──────────────────────────────────────────────────────────────────


def _tiny_patchtst(**overrides):
    cfg = {"d_model": 16, "patch_len": 8, "stride": 4, "num_layers": 1, "nhead": 2}
    cfg.update(overrides)
    return cfg


@pytest.mark.unit
def test_patchtst_forward_output_shape():
    model = PatchTST(in_channels=1, num_classes=3, **_tiny_patchtst())
    model.eval()
    out = model(torch.randn(2, 1, 64))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_patchtst_multivariate_forward():
    model = PatchTST(in_channels=5, num_classes=4, **_tiny_patchtst())
    model.eval()
    out = model(torch.randn(2, 5, 64))
    assert out.shape == (2, 4)


@pytest.mark.unit
@pytest.mark.parametrize("length", [3, 8, 37, 200])
def test_patchtst_is_length_agnostic(length):
    # Sinusoidal PE + replication padding handle T < patch_len and non-tiling T.
    model = PatchTST(in_channels=1, num_classes=3, **_tiny_patchtst())
    model.eval()
    out = model(torch.randn(2, 1, length))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_patchtst_nhead_must_divide_d_model():
    with pytest.raises(ValueError, match="divisible by nhead"):
        PatchTST(in_channels=1, num_classes=2, d_model=16, nhead=3)


@pytest.mark.unit
def test_patchtst_via_load_model():
    model = load_model(
        {
            "model": "PatchTST",
            "init_params": {"in_channels": 1, "num_classes": 2, **_tiny_patchtst()},
        },
        device="cpu",
    )
    out = model(torch.randn(2, 1, 64))
    assert out.shape == (2, 2)


# ── XLSTM ─────────────────────────────────────────────────────────────────────


def _naive_mlstm(q, k, v, igate, fgate):
    """Compute the sequential mLSTM recurrence (parallel-form ground truth)."""
    b, nh, s, dh = q.shape
    k = k / (dh**0.5)
    out = torch.zeros_like(v)
    for bi in range(b):
        for hi in range(nh):
            c_state = torch.zeros(dh, dh)
            n_state = torch.zeros(dh)
            for t in range(s):
                ig = torch.exp(igate[bi, hi, t, 0])
                fg = torch.sigmoid(fgate[bi, hi, t, 0])
                c_state = fg * c_state + ig * torch.outer(v[bi, hi, t], k[bi, hi, t])
                n_state = fg * n_state + ig * k[bi, hi, t]
                qt = q[bi, hi, t]
                den = torch.maximum((n_state @ qt).abs(), torch.tensor(1.0))
                out[bi, hi, t] = (c_state @ qt) / den
    return out


@pytest.mark.unit
def test_parallel_mlstm_matches_sequential_reference():
    # The parallel, stabilized mLSTM must equal the naive recurrence to ~1e-5.
    torch.manual_seed(0)
    b, nh, s, dh = 2, 2, 7, 4
    q, k, v = torch.randn(3, b, nh, s, dh)
    igate, fgate = torch.randn(2, b, nh, s, 1)
    fast = _parallel_mlstm(q, k, v, igate, fgate)
    ref = _naive_mlstm(q, k, v, igate, fgate)
    assert torch.allclose(fast, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_xlstm_forward_output_shape():
    model = XLSTM(in_channels=1, num_classes=3, embed_dim=16, num_blocks=1, num_heads=2)
    model.eval()
    out = model(torch.randn(2, 1, 50))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_xlstm_multivariate_forward():
    model = XLSTM(in_channels=6, num_classes=4, embed_dim=16, num_blocks=1, num_heads=2)
    model.eval()
    out = model(torch.randn(2, 6, 50))
    assert out.shape == (2, 4)


@pytest.mark.unit
@pytest.mark.parametrize("length", [5, 64])
def test_xlstm_is_length_agnostic(length):
    model = XLSTM(in_channels=1, num_classes=3, embed_dim=16, num_blocks=1, num_heads=2)
    model.eval()
    out = model(torch.randn(2, 1, length))
    assert out.shape == (2, 3)


@pytest.mark.unit
def test_xlstm_num_heads_must_divide_embed_dim():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        XLSTM(in_channels=1, num_classes=2, embed_dim=16, num_heads=3)


@pytest.mark.unit
def test_xlstm_via_load_model():
    model = load_model(
        {
            "model": "XLSTM",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "embed_dim": 16,
                "num_blocks": 1,
                "num_heads": 2,
            },
        },
        device="cpu",
    )
    out = model(torch.randn(2, 1, 50))
    assert out.shape == (2, 2)


# ── PatchTST algorithmic correctness ──────────────────────────────────────────


@pytest.mark.unit
def test_patchtst_channel_permutation_invariance():
    # Channel-independent encoding + mean over channels => permuting channels
    # must leave the output unchanged (exact up to float).
    torch.manual_seed(0)
    model = PatchTST(in_channels=4, num_classes=3, **_tiny_patchtst())
    model.eval()
    x = torch.randn(2, 4, 64)
    out = model(x)
    permuted = model(x[:, [2, 0, 3, 1], :])
    assert torch.allclose(out, permuted, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("length", "expected"), [(5, 1), (8, 1), (16, 3), (18, 4), (20, 4)]
)
def test_patchtst_patch_count(length, expected):
    # patch_len=8, stride=4: padding tiles T into floor((T_pad-8)/4)+1 patches,
    # with T<patch_len padded up to a single patch. Captured via the embed input.
    model = PatchTST(in_channels=1, num_classes=2, **_tiny_patchtst())
    model.eval()
    captured = {}
    handle = model.patch_embed.register_forward_hook(
        lambda _m, inp, _out: captured.update(n=inp[0].shape[1])
    )
    model(torch.randn(1, 1, length))
    handle.remove()
    assert captured["n"] == expected


@pytest.mark.unit
def test_patchtst_positional_encoding_formula():
    pe = PatchTST._positional_encoding(4, 6, torch.device("cpu"), torch.float32)
    assert pe.shape == (1, 4, 6)
    # Position 0: sin(0)=0 at even indices, cos(0)=1 at odd indices.
    assert torch.allclose(pe[0, 0, 0::2], torch.zeros(3), atol=1e-6)
    assert torch.allclose(pe[0, 0, 1::2], torch.ones(3), atol=1e-6)


# ── Cross-model gradient flow ─────────────────────────────────────────────────

# Tiny, fast-to-build configs for every registered built-in model.
_TINY_MODEL_CONFIGS = [
    {"model": "FCN", "init_params": {"filters": [8, 16, 8], "kernel_sizes": [7, 5, 3]}},
    {"model": "LeNet", "init_params": {"head_hidden": 8}},
    {"model": "ResNet", "init_params": {"n_feature_maps": 8}},
    {"model": "LSTM", "init_params": {"hidden_size": 16, "num_layers": 1}},
    {
        "model": "PatchTST",
        "init_params": {
            "d_model": 16,
            "patch_len": 8,
            "stride": 4,
            "num_layers": 1,
            "nhead": 2,
        },
    },
    {
        "model": "XLSTM",
        "init_params": {"embed_dim": 16, "num_blocks": 1, "num_heads": 2},
    },
]
_MODEL_IDS = [c["model"] for c in _TINY_MODEL_CONFIGS]


@pytest.mark.unit
@pytest.mark.parametrize("config", _TINY_MODEL_CONFIGS, ids=_MODEL_IDS)
def test_gradients_reach_all_parameters(config):
    # A backward pass must produce a gradient for every parameter (graph is fully
    # connected — no detached/dead subgraph) and a non-zero total gradient.
    torch.manual_seed(0)
    model = load_model(
        {
            "model": config["model"],
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                **config["init_params"],
            },
        },
        device="cpu",
    )
    model.eval()  # deterministic connectivity (no dropout masking grads to None)
    model(torch.randn(4, 1, 64)).sum().backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"{config['model']} parameters without grad: {missing}"
    total = sum(p.grad.abs().sum() for p in model.parameters())
    assert total > 0


@pytest.mark.unit
@pytest.mark.parametrize("config", [_TINY_LENET, _TINY_RESNET])
def test_wagner_models_save_then_load_round_trip(tmp_path, config):
    original = load_model(config, device="cpu", save_path=tmp_path)
    save_model(original, ckpt=config, save_path=tmp_path)

    reloaded = load_model(
        {**config, "model_ckpt_path": str(tmp_path / "model_dict.pt")},
        device="cpu",
        save_path=tmp_path,
    )

    orig_state = original.state_dict()
    new_state = reloaded.state_dict()
    assert orig_state.keys() == new_state.keys()
    for key, tensor in orig_state.items():
        assert torch.equal(tensor, new_state[key])


# ── load_model unsupported checkpoint formats ────────────────────────────────────


@pytest.mark.unit
def test_load_model_h5_returns_none_with_info(tmp_path, caplog):
    ckpt = tmp_path / "model.h5"
    ckpt.write_bytes(b"not a real checkpoint")
    with caplog.at_level("INFO", logger="xai4tsc.models.models"):
        model = load_model({"model": "x", "model_ckpt_path": str(ckpt)})
    assert model is None
    assert "not implemented" in caplog.text.lower()


@pytest.mark.unit
def test_load_model_ckpt_returns_none_with_info(tmp_path, caplog):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"not a real checkpoint")
    with caplog.at_level("INFO", logger="xai4tsc.models.models"):
        model = load_model({"model": "x", "model_ckpt_path": str(ckpt)})
    assert model is None
    assert "not implemented" in caplog.text.lower()


# ── load_model: bare state-dict checkpoints (R22) ─────────────────────────────


def _bare_checkpoint(tmp_path, config, name="FCN_epoch_3.pt", epoch=3):
    """Train-free stand-in for what the runner writes: a checkpoint + its config."""
    model = load_model(config, device="cpu", save_path=tmp_path)
    model.save_model(tmp_path / name, epoch=epoch)
    return model, tmp_path / name


@pytest.mark.unit
@pytest.mark.parametrize("config", [_TINY_FCN, _TINY_LENET])
def test_load_model_reloads_bare_state_dict_via_registry(tmp_path, config):
    original, ckpt = _bare_checkpoint(tmp_path, config)

    reloaded = load_model(
        {**config, "model_ckpt_path": str(ckpt)}, device="cpu", save_path=tmp_path
    )

    assert type(reloaded) is type(original)
    orig_state = original.state_dict()
    assert orig_state.keys() == reloaded.state_dict().keys()
    for key, tensor in orig_state.items():
        assert torch.equal(tensor, reloaded.state_dict()[key])


@pytest.mark.unit
def test_load_model_reads_init_params_from_config_sidecar(tmp_path):
    """Only ``model`` + ``model_ckpt_path`` is enough once the sidecar exists."""
    original, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)

    reloaded = load_model({"model": "FCN", "model_ckpt_path": str(ckpt)}, device="cpu")

    assert reloaded._init_params == _TINY_FCN["init_params"]
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[key])


@pytest.mark.unit
def test_load_model_sidecar_overrides_mismatched_config(tmp_path):
    """The architecture the weights fit wins over a config written elsewhere."""
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)

    # A reload config carrying another dataset's auto-detected values.
    wrong = {
        "model": "FCN",
        "model_ckpt_path": str(ckpt),
        "init_params": {"in_channels": 7, "num_classes": 5},
    }
    reloaded = load_model(wrong, device="cpu")

    assert reloaded._init_params["in_channels"] == 1
    assert reloaded._init_params["num_classes"] == 2


@pytest.mark.unit
def test_load_model_sidecar_nulls_do_not_override_config(tmp_path):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)
    with open(ModelBase.config_path_for(ckpt), "w") as f:
        json.dump({"model": None, "init_params": {"num_classes": None}}, f)

    reloaded = load_model(
        {**_TINY_FCN, "model_ckpt_path": str(ckpt)}, device="cpu", save_path=tmp_path
    )

    assert reloaded.name == "FCN"
    assert reloaded._init_params["num_classes"] == 2


@pytest.mark.unit
def test_load_model_missing_checkpoint_names_path_and_siblings(tmp_path):
    _bare_checkpoint(tmp_path, _TINY_FCN)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_model(
            {**_TINY_FCN, "model_ckpt_path": str(tmp_path / "model_dict.pt")},
            device="cpu",
        )

    message = str(excinfo.value)
    assert "model_dict.pt" in message
    assert "FCN_epoch_3.pt" in message


@pytest.mark.unit
def test_load_model_unresolvable_checkpoint_names_what_was_tried(tmp_path):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN, name="orphan.pt")
    ModelBase.config_path_for(ckpt).unlink(missing_ok=True)

    with pytest.raises(ValueError) as excinfo:
        load_model({"model": "not_registered", "model_ckpt_path": str(ckpt)})

    message = str(excinfo.value)
    assert "orphan.pt" in message
    assert "not_registered" in message
    assert "model_class" in message


# ── the per-checkpoint config file (R22) ──────────────────────────────────────


@pytest.mark.unit
def test_config_path_for_is_named_after_its_checkpoint():
    assert ModelBase.config_path_for(Path("/r/FCN_epoch_10.pt")).name == (
        "FCN_epoch_10_config.json"
    )
    assert ModelBase.config_path_for("/r/model_dict.pt").name == (
        "model_dict_config.json"
    )


@pytest.mark.unit
def test_save_model_writes_the_checkpoint_config_beside_it(tmp_path):
    model = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)

    model.save_model(tmp_path / "FCN_epoch_7.pt", epoch=7)

    written = json.loads((tmp_path / "FCN_epoch_7_config.json").read_text())
    assert written == {
        "model": "FCN",
        "init_params": _TINY_FCN["init_params"],
        "epoch": 7,
    }
    # Not resumed, so the key is absent rather than null.
    assert "resumed_from" not in written


@pytest.mark.unit
def test_checkpoint_config_records_unknown_epoch_as_null(tmp_path):
    model = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)

    model.save_model(tmp_path / "weights.pt")

    written = json.loads((tmp_path / "weights_config.json").read_text())
    assert written["epoch"] is None, "an unknown history must not become a guessed 0"


@pytest.mark.unit
def test_checkpoint_config_records_resumed_from_when_resumed(tmp_path):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)
    resumed = load_model(
        {"model": "FCN", "model_ckpt_path": str(ckpt)}, device="cpu", save_path=tmp_path
    )

    resumed.save_model(tmp_path / "FCN_epoch_5.pt", epoch=5)

    written = json.loads((tmp_path / "FCN_epoch_5_config.json").read_text())
    assert written["resumed_from"] == str(ckpt)
    assert written["epoch"] == 5


@pytest.mark.unit
def test_checkpoint_config_skipped_for_unnamed_model(tmp_path):
    """A directly-constructed model has no name worth recording (see R28)."""
    model = _TinyModel()

    model.save_model(tmp_path / "x_epoch_1.pt")

    assert not (tmp_path / "x_epoch_1_config.json").exists()


# ── load_model logging and epoch provenance ───────────────────────────────────

_LOGGER = "xai4tsc.models.models"


@pytest.mark.unit
def test_load_model_logs_the_checkpoint_path_and_registry_origin(tmp_path, caplog):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)

    with caplog.at_level("INFO", logger=_LOGGER):
        load_model({**_TINY_FCN, "model_ckpt_path": str(ckpt)}, device="cpu")

    assert str(ckpt) in caplog.text
    assert "class from the MODELS registry" in caplog.text
    assert "trained 3 epoch(s)" in caplog.text


@pytest.mark.unit
def test_load_model_logs_model_class_origin(tmp_path, caplog):
    original = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)
    save_model(original, ckpt=_TINY_FCN, save_path=tmp_path)

    with caplog.at_level("INFO", logger=_LOGGER):
        load_model(
            {**_TINY_FCN, "model_ckpt_path": str(tmp_path / "model_dict.pt")},
            device="cpu",
        )

    assert "class from the checkpoint's 'model_class'" in caplog.text


@pytest.mark.unit
def test_load_model_logs_when_building_without_a_checkpoint(caplog):
    with caplog.at_level("INFO", logger=_LOGGER):
        load_model(_TINY_FCN, device="cpu")

    assert "Built FCN from the MODELS registry" in caplog.text


@pytest.mark.unit
def test_load_model_logs_what_the_config_file_supplied(tmp_path, caplog):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)

    with caplog.at_level("INFO", logger=_LOGGER):
        load_model({"model": "FCN", "model_ckpt_path": str(ckpt)}, device="cpu")

    assert "FCN_epoch_3_config.json supplied" in caplog.text
    assert "take precedence" in caplog.text


@pytest.mark.unit
def test_load_model_warns_when_the_config_overrides_a_given_value(tmp_path, caplog):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)
    mismatched = {
        "model": "FCN",
        "model_ckpt_path": str(ckpt),
        "init_params": {**_TINY_FCN["init_params"], "num_classes": 5},
    }

    with caplog.at_level("WARNING", logger=_LOGGER):
        load_model(mismatched, device="cpu")

    assert "overrides configured init_params.num_classes: 5 -> 2" in caplog.text


@pytest.mark.unit
def test_load_model_seeds_best_epoch_from_the_config_file(tmp_path):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN, name="weights.pt", epoch=12)

    model = load_model({**_TINY_FCN, "model_ckpt_path": str(ckpt)}, device="cpu")

    assert model.best_epoch == 12, "the config file is the primary source"
    assert model._loaded_from == ckpt


@pytest.mark.unit
def test_load_model_falls_back_to_the_epoch_in_the_filename(tmp_path):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN, name="FCN_epoch_9.pt")
    ModelBase.config_path_for(ckpt).unlink()

    model = load_model({**_TINY_FCN, "model_ckpt_path": str(ckpt)}, device="cpu")

    assert model.best_epoch == 9


@pytest.mark.unit
def test_load_model_warns_when_the_epoch_cannot_be_inferred(tmp_path, caplog):
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN, name="weights.pt")
    ModelBase.config_path_for(ckpt).unlink()

    with caplog.at_level("WARNING", logger=_LOGGER):
        model = load_model({**_TINY_FCN, "model_ckpt_path": str(ckpt)}, device="cpu")

    assert model.best_epoch is None
    assert "Cannot tell how many epochs" in caplog.text
    assert "numbering restarts at 1" in caplog.text


@pytest.mark.unit
def test_load_model_honours_a_legacy_directory_level_config(tmp_path):
    """Trees written before per-checkpoint configs must still load."""
    _, ckpt = _bare_checkpoint(tmp_path, _TINY_FCN)
    ModelBase.config_path_for(ckpt).unlink()
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"model": "FCN", "init_params": _TINY_FCN["init_params"]}, f)

    model = load_model({"model": "FCN", "model_ckpt_path": str(ckpt)}, device="cpu")

    assert model._init_params == _TINY_FCN["init_params"]


# ── resuming continues the epoch count ────────────────────────────────────────

_TRAIN_HP = {
    "epochs": 2,
    "batchsize": 8,
    "learn_rate": 1e-3,
    "patience": 5,
    "save_best": True,
    "loss_func": "crossentropy",
    "optimizer": "adam",
}


def _tiny_batch():
    import numpy as np

    return (
        np.random.default_rng(0).standard_normal((16, 1, 32)).astype("float32"),
        np.array([0, 1] * 8),
    )


@pytest.mark.unit
def test_training_from_scratch_numbers_epochs_from_one(tmp_path):
    model = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)
    data, labels = _tiny_batch()

    model.train_model(data, labels, **_TRAIN_HP)

    # The surviving checkpoint is the *best* epoch, which may be 1 or 2 - the
    # point is that it is numbered within this call's budget.
    assert model.best_epoch in (1, 2)
    assert [f.name for f in tmp_path.glob("*.pt")] == [
        f"FCN_epoch_{model.best_epoch}.pt"
    ]


@pytest.mark.unit
def test_resumed_training_continues_the_epoch_count(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    model = load_model(_TINY_FCN, device="cpu", save_path=first)
    data, labels = _tiny_batch()
    model.train_model(data, labels, **_TRAIN_HP)
    checkpoint = next(first.glob("FCN_epoch_*.pt"))

    second = tmp_path / "second"
    resumed = load_model(
        {**_TINY_FCN, "model_ckpt_path": str(checkpoint)},
        device="cpu",
        save_path=second,
    )
    resumed.train_model(data, labels, **_TRAIN_HP)

    assert resumed.best_epoch > 2, "a resumed run must not restart the epoch count"
    assert [f.name for f in second.glob("*.pt")] == [
        f"FCN_epoch_{resumed.best_epoch}.pt"
    ]
    # Exactly one checkpoint/config pair survives: superseded ones are removed.
    assert len(list(second.glob("*_config.json"))) == 1


@pytest.mark.unit
def test_explicit_start_epoch_overrides_the_checkpoint(tmp_path):
    model = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)
    data, labels = _tiny_batch()

    model.train_model(data, labels, **_TRAIN_HP, start_epoch=100)

    assert model.best_epoch in (101, 102)
    assert list(tmp_path.glob("FCN_epoch_10*.pt"))


@pytest.mark.unit
@pytest.mark.parametrize("model_class", list(MODELS.values()))
def test_every_builtin_records_json_safe_init_params(model_class):
    # The sidecar is JSON, and a value it cannot encode means no sidecar at all.
    # locals() also carries the __class__ cell that a zero-arg super() puts in
    # scope, which is not encodable - hence the check rather than a spot test.
    model = model_class(in_channels=1, num_classes=2)

    assert model._init_params, f"{model_class.__name__} records nothing"
    assert "self" not in model._init_params
    assert not [k for k in model._init_params if k.startswith("__")]
    json.dumps(model._init_params)


@pytest.mark.unit
def test_recorded_params_include_defaults_not_only_what_was_passed():
    # Recording the defaults too is what lets an old checkpoint rebuild the same
    # architecture after a default changes in a later version.
    model = FCN(in_channels=1, num_classes=2)

    assert model._init_params == {
        "in_channels": 1,
        "num_classes": 2,
        "filters": (128, 256, 128),
        "kernel_sizes": (8, 5, 3),
    }


@pytest.mark.unit
def test_configure_sets_operational_attrs_and_chains():
    model = FCN(in_channels=1, num_classes=2).configure(
        device="cpu", save_path="/tmp/does-not-need-to-exist", name="Renamed"
    )

    assert model.name == "Renamed"
    assert model.device == torch.device("cpu")
    assert model.save_path == Path("/tmp/does-not-need-to-exist")
    # Architecture record is the constructor's business, untouched by configure.
    assert model._init_params["in_channels"] == 1


@pytest.mark.unit
def test_configure_leaves_omitted_attrs_alone():
    # Otherwise a later call that sets one attribute silently clears the others.
    model = FCN(in_channels=1, num_classes=2).configure(
        device="cpu", save_path="/tmp/first"
    )

    model.configure(name="Second")

    assert model.name == "Second"
    assert model.save_path == Path("/tmp/first")
    assert model.device == torch.device("cpu")


@pytest.mark.unit
def test_unencodable_init_params_skip_the_sidecar_and_name_the_key(tmp_path, caplog):
    # A directly built model can be handed anything, unlike a YAML config. Say
    # which argument is the problem instead of writing a truncated config.
    class _Unencodable(ModelBase):
        def __init__(self, in_channels: int = 1, transform=None):
            super().__init__()
            self._record_init_params(locals())
            self.fc = nn.Linear(in_channels, 2)

        def forward(self, x):
            return self.fc(x.mean(-1))

    model = _Unencodable(transform=lambda v: v)

    with caplog.at_level("WARNING", logger="xai4tsc.models.base"):
        model.save_model(tmp_path / "_Unencodable_epoch_1.pt", epoch=1)

    assert not list(tmp_path.glob("*_config.json"))
    assert "transform" in caplog.text
    assert (tmp_path / "_Unencodable_epoch_1.pt").exists(), "weights still saved"


@pytest.mark.unit
@pytest.mark.parametrize("method", ["train_model", "evaluate_model"])
def test_writing_without_a_destination_raises(method):
    # Falling back to the working directory used to scatter checkpoints and
    # plots wherever the caller happened to be standing.
    model = FCN(in_channels=1, num_classes=2)
    data, labels = _tiny_batch()

    # evaluate_model takes only the batch size; the rest describes training.
    kwargs = _TRAIN_HP if method == "train_model" else {"batchsize": 8}
    with pytest.raises(ValueError, match="needs a save_path"):
        getattr(model, method)(data, labels, **kwargs)


@pytest.mark.unit
def test_directly_built_model_writes_files_named_after_its_class(tmp_path):
    # The regression guard for the "None_*" litter: a model that never went
    # through load_model still names every artifact it writes.
    model = FCN(in_channels=1, num_classes=2)
    data, labels = _tiny_batch()

    model.train_model(data, labels, **_TRAIN_HP, save_path=tmp_path)
    model.evaluate_model(data, labels, batchsize=8)

    written = sorted(f.name for f in tmp_path.iterdir())
    assert not [f for f in written if f.startswith("None")], written
    assert f"FCN_epoch_{model.best_epoch}.pt" in written
    assert "FCN_training_curve.png" in written
    # The constructor records its own arguments, so a directly built model does
    # get a sidecar and its checkpoint reloads from the path alone.
    assert len(list(tmp_path.glob("*_config.json"))) == 1


# ── predict_probs ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_predict_probs_returns_a_distribution():
    model = load_model(_TINY_FCN, device="cpu")
    data = np.random.randn(5, 1, 16).astype(np.float32)

    probs = predict_probs(model, data)

    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


@pytest.mark.unit
def test_predict_probs_matches_model_base_predict():
    """ModelBase.predict delegates, so the two must agree exactly."""
    model = load_model(_TINY_FCN, device="cpu")
    data = np.random.randn(4, 1, 16).astype(np.float32)

    _, via_method = model.predict(data)

    assert np.array_equal(predict_probs(model, data), via_method)


@pytest.mark.unit
def test_predict_probs_works_on_a_bare_nn_module():
    """The point of the helper: no ModelBase, so no .predict to call."""
    bare = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))

    probs = predict_probs(bare, np.random.randn(2, 1, 16).astype(np.float32))

    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.unit
def test_predict_probs_leaves_train_mode_untouched():
    """The model may belong to the caller — a metric re-explaining, say."""
    model = load_model(_TINY_FCN, device="cpu")
    data = np.random.randn(2, 1, 16).astype(np.float32)

    model.train()
    predict_probs(model, data)
    assert model.training, "predict_probs must not switch the model to eval"

    model.eval()
    predict_probs(model, data)
    assert not model.training


@pytest.mark.unit
def test_predict_probs_handles_a_parameterless_module():
    """No parameters means no device to move to; the input is used as-is."""

    class _Constant(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2)

    probs = predict_probs(_Constant(), np.zeros((3, 1, 8), dtype=np.float32))

    assert np.allclose(probs, 0.5)


@pytest.mark.unit
def test_predict_probs_accepts_tensor_and_ndarray_alike():
    """One entry point serves callers holding either; evaluate_model passes tensors."""
    model = load_model(_TINY_FCN, device="cpu")
    array = np.random.randn(3, 1, 16).astype(np.float32)

    from_array = predict_probs(model, array)
    from_tensor = predict_probs(model, torch.from_numpy(array))

    assert np.array_equal(from_array, from_tensor)
