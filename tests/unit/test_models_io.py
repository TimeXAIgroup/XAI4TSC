"""Unit tests for model persistence/registry — light torch use, no training."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from xai4tsc.models.base import (
    ModelBase,
    _confusion_counts,
    _macro,
    _rates,
    predict_probs,
)
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

# What the model RECORDS: what was passed, plus the constructor defaults it did
# not get. Keeping the two separate is the point of
# test_recorded_params_include_defaults_not_only_what_was_passed.
_TINY_FCN_RECORDED = _TINY_FCN["init_params"] | {"output_activation": "softmax"}


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
    assert model.save_path == tmp_path.resolve()
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

    assert reloaded._init_params == _TINY_FCN_RECORDED
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
        "init_params": _TINY_FCN_RECORDED,
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

    assert model._init_params == _TINY_FCN_RECORDED


# ── resuming continues the epoch count ────────────────────────────────────────

_TRAIN_HP = {
    "epochs": 2,
    "batchsize": 8,
    "learn_rate": 1e-3,
    "patience": 5,
    "save_best": True,
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
        "output_activation": "softmax",
    }


@pytest.mark.unit
def test_configure_sets_operational_attrs_and_chains():
    model = FCN(in_channels=1, num_classes=2).configure(
        device="cpu", save_path="/tmp/does-not-need-to-exist", name="Renamed"
    )

    assert model.name == "Renamed"
    assert model.device == torch.device("cpu")
    # save_path is stored absolute, which is not the same string where /tmp is
    # a symlink (macOS), so compare the resolved form.
    assert model.save_path == Path("/tmp/does-not-need-to-exist").resolve()
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
    assert model.save_path == Path("/tmp/first").resolve()
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
    with pytest.raises(ValueError, match="No save_path set"):
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


# ── test result artifacts: per-class counts, and pos/neg only where they mean ─
# something


@pytest.mark.unit
def test_summary_reports_pos_neg_only_for_a_binary_model(tmp_path):
    # pos/neg are sum(y_true == 1) and sum(y_true == 0): the class balance for a
    # binary problem, and for an 8-class one two arbitrary class counts that read
    # as a balance. The guarded log line always omitted them; the CSV did not.
    data = np.random.default_rng(0).standard_normal((12, 1, 32)).astype("float32")

    binary = FCN(in_channels=1, num_classes=2).evaluate_model(
        data, np.array([0, 1] * 6), batchsize=4, save_path=tmp_path / "binary"
    )
    multi = FCN(in_channels=1, num_classes=3).evaluate_model(
        data, np.array([0, 1, 2] * 4), batchsize=4, save_path=tmp_path / "multi"
    )

    assert {"pos", "neg"} <= set(binary)
    assert not {"pos", "neg"} & set(multi)
    assert {"macro_f1", "macro_mcc"} <= set(multi)
    # The CSV is written straight from the dict, which is how its columns drifted
    # away from the log line in the first place.
    for returned, directory in ((binary, "binary"), (multi, "multi")):
        csv = next((tmp_path / directory).glob("*_test_results.csv"))
        assert csv.read_text().splitlines()[0].split(",") == list(returned)


@pytest.mark.unit
def test_per_class_results_are_written_for_every_problem(tmp_path):
    # The confusion counts are computed to draw the matrix anyway; a sibling file
    # keeps them readable for a 60-class dataset, where one wide row would not.
    data = np.random.default_rng(1).standard_normal((12, 1, 32)).astype("float32")
    labels = np.array([0, 1, 2] * 4)

    FCN(in_channels=1, num_classes=3).evaluate_model(
        data, labels, batchsize=4, save_path=tmp_path, class_names=["a", "b", "c"]
    )

    csv = next(tmp_path.glob("*_test_per_class.csv"))
    rows = [line.split(",") for line in csv.read_text().strip().splitlines()]
    assert rows[0][:7] == ["class", "name", "samples", "tp", "tn", "fp", "fn"]
    assert [r[0] for r in rows[1:]] == ["0", "1", "2"]
    assert [r[1] for r in rows[1:]] == ["a", "b", "c"]
    assert [int(r[2]) for r in rows[1:]] == [4, 4, 4]
    # tp + tn + fp + fn is the whole test set for each class, one-vs-rest.
    assert all(sum(int(v) for v in r[3:7]) == 12 for r in rows[1:])


@pytest.mark.unit
def test_the_per_class_table_reports_each_class_its_own_accuracy(tmp_path):
    # The summary's accuracy is the whole split's; per class the one-vs-rest
    # reading is the only one available, and it was the column the per-output
    # table had and this one did not.
    data = np.random.default_rng(1).standard_normal((12, 1, 32)).astype("float32")
    labels = np.array([0, 1, 2] * 4)

    FCN(in_channels=1, num_classes=3).evaluate_model(
        data, labels, batchsize=4, save_path=tmp_path
    )

    csv = next(tmp_path.glob("*_test_per_class.csv"))
    header, *body = csv.read_text().strip().splitlines()
    cols = header.split(",")
    assert cols.index("accuracy") == cols.index("fn") + 1, "with the other rates"
    for line in body:
        cell = dict(zip(cols, line.split(","), strict=True))
        counts = (int(cell["tp"]), int(cell["tn"]))
        assert float(cell["accuracy"]) == pytest.approx(sum(counts) / 12, abs=1e-4)


@pytest.mark.unit
def test_both_formulations_report_the_same_per_class_columns(tmp_path):
    # Single-label and multi-label write the same filename, so a reader that
    # handles one has to handle the other. Both derive their rates from _rates,
    # which is what keeps the columns identical.
    data = np.random.default_rng(2).standard_normal((12, 1, 32)).astype("float32")

    FCN(in_channels=1, num_classes=3).evaluate_model(
        data, np.array([0, 1, 2] * 4), batchsize=4, save_path=tmp_path / "single"
    )
    FCN(in_channels=1, num_classes=3, output_activation="sigmoid").evaluate_model(
        data,
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]] * 3),
        batchsize=4,
        save_path=tmp_path / "multi",
    )

    def header(directory: str) -> list[str]:
        csv = next((tmp_path / directory).glob("*_test_per_class.csv"))
        return csv.read_text().splitlines()[0].split(",")

    assert header("single") == header("multi")


@pytest.mark.unit
def test_class_names_of_the_wrong_length_are_reported_and_ignored(tmp_path, caplog):
    data = np.random.default_rng(2).standard_normal((8, 1, 32)).astype("float32")

    with caplog.at_level("WARNING", logger="xai4tsc.models.base"):
        FCN(in_channels=1, num_classes=2).evaluate_model(
            data,
            np.array([0, 1] * 4),
            batchsize=4,
            save_path=tmp_path,
            class_names=["only", "three", "names"],
        )

    assert "Ignoring 3 class names" in caplog.text
    csv = next(tmp_path.glob("*_test_per_class.csv"))
    assert [line.split(",")[1] for line in csv.read_text().splitlines()[1:]] == [
        "0",
        "1",
    ]


@pytest.mark.unit
def test_confusion_counts_are_indexed_by_class_not_by_presence():
    # The full matrix is what makes samples_k mean class k. Built over only the
    # classes present, an absent class shifts every row after it. (evaluate_model
    # cannot be driven this way yet: its multiclass AUC rejects a test split that
    # misses a class, so the guarantee is checked on the helper.)
    cm = np.array([[3, 1, 0], [2, 4, 0], [0, 0, 0]])  # class 2 absent

    counts = _confusion_counts(cm)

    assert len(counts) == 3
    assert counts[2] == {"samples": 0, "tp": 0, "tn": 10, "fp": 0, "fn": 0}
    assert counts[0] == {"samples": 4, "tp": 3, "tn": 4, "fp": 2, "fn": 1}
    assert all(c["tp"] + c["tn"] + c["fp"] + c["fn"] == 10 for c in counts)


@pytest.mark.unit
def test_rates_are_nan_where_undefined_rather_than_raising():
    # A class with no true samples and none predicted divides by zero in every
    # rate; the evaluation must still produce a row for it.
    empty = _rates(tp=0, tn=10, fp=0, fn=0)
    assert np.isnan(empty["sensitivity"]) and np.isnan(empty["f1"])
    assert empty["specificity"] == 1.0
    # Accuracy is the exception: a class nothing belongs to and nothing predicts
    # is right about every sample, so it is 1.0 rather than nan.
    assert empty["accuracy"] == 1.0
    assert np.isnan(_rates(tp=0, tn=0, fp=0, fn=0)["accuracy"]), "no samples at all"

    perfect = _rates(tp=5, tn=5, fp=0, fn=0)
    assert perfect == pytest.approx(
        {
            "accuracy": 1.0,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "ppv": 1.0,
            "npv": 1.0,
            "f1": 1.0,
            "mcc": 1.0,
        }
    )


@pytest.mark.unit
def test_macro_averages_over_the_classes_that_define_a_rate():
    # One degenerate class must not turn the whole summary into nan.
    rates = [{"f1": 1.0}, {"f1": 0.5}, {"f1": float("nan")}]

    assert _macro(rates, "f1") == pytest.approx(0.75)
    assert np.isnan(_macro([{"f1": float("nan")}], "f1"))


# ── the ModelBase contract: validation data, save_path, degenerate splits ─────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "missing"),
    [({"data_val": True}, "labels_val"), ({"labels_val": True}, "data_val")],
)
def test_half_a_validation_set_raises(tmp_path, kwargs, missing):
    # Dropped silently, the run selects its checkpoint on training loss while
    # the caller believes it is validating - the one thing the method's own
    # docstring warns against.
    data, labels = _tiny_batch()
    supplied = {k: (data if k == "data_val" else labels) for k in kwargs}

    with pytest.raises(ValueError, match=f"without {missing}"):
        FCN(in_channels=1, num_classes=2).train_model(
            data, labels, **_TRAIN_HP, save_path=tmp_path, **supplied
        )


@pytest.mark.unit
def test_an_empty_validation_set_raises(tmp_path):
    data, labels = _tiny_batch()

    with pytest.raises(ValueError, match="empty validation set"):
        FCN(in_channels=1, num_classes=2).train_model(
            data,
            labels,
            **_TRAIN_HP,
            save_path=tmp_path,
            data_val=data[:0],
            labels_val=labels[:0],
        )


@pytest.mark.unit
def test_training_without_validation_data_says_what_it_monitors(tmp_path, caplog):
    # A deliberate no-validation run is supported; the point is that it leaves a
    # trace in the log instead of only an absent val_loss.
    data, labels = _tiny_batch()

    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        FCN(in_channels=1, num_classes=2).train_model(
            data, labels, **_TRAIN_HP, save_path=tmp_path
        )

    assert caplog.text.count("monitor TRAINING loss") == 1
    assert "val_loss" not in caplog.text


@pytest.mark.unit
def test_a_real_validation_set_still_drives_selection(tmp_path, caplog):
    data, labels = _tiny_batch()

    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        FCN(in_channels=1, num_classes=2).train_model(
            data,
            labels,
            **_TRAIN_HP,
            save_path=tmp_path,
            data_val=data,
            labels_val=labels,
        )

    assert "val_loss" in caplog.text
    assert "monitor TRAINING loss" not in caplog.text


@pytest.mark.unit
def test_evaluating_elsewhere_does_not_move_the_models_home(tmp_path):
    # A one-off destination used to become the model's default for every later
    # call, so evaluating into two directories left it pointing at the second.
    home, other = tmp_path / "home", tmp_path / "other"
    data, labels = _tiny_batch()
    model = FCN(in_channels=1, num_classes=2)
    model.train_model(data, labels, **_TRAIN_HP, save_path=home)
    # Resolved, so the comparison holds where the temp dir is reached through a
    # symlink (macOS /var -> /private/var).
    assert model.save_path == home.resolve(), "training establishes the home"

    model.evaluate_model(data, labels, batchsize=8, save_path=other)

    assert model.save_path == home.resolve()
    assert list(other.glob("*_test_results.csv")), "it wrote where it was told"
    # ...and with no argument at all, back to the home training established.
    model.evaluate_model(data, labels, batchsize=8)
    assert list(home.glob("*_test_results.csv"))


@pytest.mark.unit
def test_a_class_absent_from_the_test_split_scores_partially(tmp_path):
    # The multiclass AUC used to hand sklearn the probability columns of the
    # present classes only, which no longer sum to 1, so it raised - and the
    # "(partial, k/K classes)" label the code builds was unreachable.
    data = np.random.default_rng(3).standard_normal((15, 1, 32)).astype("float32")
    labels = np.array([0, 1, 2] * 5)  # class 3 never appears

    results = FCN(in_channels=1, num_classes=4).evaluate_model(
        data, labels, batchsize=5, save_path=tmp_path
    )

    assert "partial, 3/4 classes" in results["auc"]
    rows = [
        line.split(",")
        for line in next(tmp_path.glob("*_test_per_class.csv")).read_text().splitlines()
    ]
    header, body = rows[0], rows[1:]
    assert len(body) == 4, "one row per model class, present or not"
    absent = dict(zip(header, body[3], strict=True))
    assert int(absent["samples"]) == 0
    assert absent["auc"] == "nan", "no AUC for a class with no samples"


@pytest.mark.unit
def test_the_multiclass_auc_is_sklearns_ovr_macro(tmp_path):
    # The guarantee that replacing the sklearn call moved no recorded number.
    data = np.random.default_rng(4).standard_normal((18, 1, 32)).astype("float32")
    labels = np.array([0, 1, 2] * 6)
    model = FCN(in_channels=1, num_classes=3)

    results = model.evaluate_model(data, labels, batchsize=6, save_path=tmp_path)

    expected = roc_auc_score(
        labels,
        predict_probs(model, data).astype(np.float64),
        multi_class="ovr",
        labels=np.arange(3),
    )
    assert results["auc"] == f"{expected:.4f}"


@pytest.mark.unit
@pytest.mark.parametrize("num_classes", [2, 3])
def test_a_single_class_test_split_completes_without_a_roc(tmp_path, num_classes):
    # Both used to crash here: the multiclass path on an unbound y_prob, the
    # binary one on unique_classes[1]. There is no curve to draw, but the counts
    # and the matrix are still worth writing.
    data = np.random.default_rng(5).standard_normal((6, 1, 32)).astype("float32")
    labels = np.zeros(6, dtype=np.int64)

    results = FCN(in_channels=1, num_classes=num_classes).evaluate_model(
        data, labels, batchsize=3, save_path=tmp_path
    )

    assert results["auc"] == "nan"
    assert list(tmp_path.glob("*_test_results.csv"))
    assert list(tmp_path.glob("*_test_per_class.csv"))
    assert list(tmp_path.glob("*_confusion_matrix.png"))
    assert not list(tmp_path.glob("*_roc.png")), "no curve exists to plot"


@pytest.mark.unit
@pytest.mark.parametrize("num_classes", [2, 3])
def test_a_degenerate_split_keeps_the_columns_of_its_problem_type(
    tmp_path, num_classes
):
    # Only the AUC needs two classes present; every rate is nan-safe arithmetic
    # over the confusion counts. The binary path used to skip the whole rate
    # block here and emit a short row, so one degenerate dataset in a benchmark
    # left half of models.csv empty for every other row too.
    data = np.random.default_rng(8).standard_normal((12, 1, 32)).astype("float32")
    model = FCN(in_channels=1, num_classes=num_classes)
    labels = np.arange(12) % num_classes

    healthy = model.evaluate_model(
        data, labels, batchsize=4, save_path=tmp_path / "healthy"
    )
    degenerate = model.evaluate_model(
        data, np.zeros(12, dtype=np.int64), batchsize=4, save_path=tmp_path / "one"
    )

    assert list(degenerate) == list(healthy)
    assert degenerate["auc"] == "nan", "the AUC is the only thing lost"
    assert degenerate["accuracy"] != "nan", "accuracy needs no second class"


# ── output_activation ─────────────────────────────────────────────────────────

# predict_probs is the single probability path for prediction, evaluation and
# every explainer that picks a target, so which activation it applies decides a
# lot of numbers. It is a CONSTRUCTOR argument because only init_params are
# persisted to a checkpoint: anywhere else and a reloaded multi-label model gets
# softmaxed with nothing in the checkpoint to catch it.


@pytest.mark.unit
def test_predict_probs_softmaxes_by_default():
    model = FCN(in_channels=1, num_classes=3).eval()
    probs = predict_probs(model, np.zeros((4, 1, 32), dtype=np.float32))
    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)


@pytest.mark.unit
def test_predict_probs_applies_sigmoid_when_the_model_declares_it():
    model = FCN(in_channels=1, num_classes=3, output_activation="sigmoid").eval()
    x = np.zeros((4, 1, 32), dtype=np.float32)

    probs = predict_probs(model, x)

    assert probs.shape == (4, 3)
    assert ((probs >= 0) & (probs <= 1)).all()
    # The contract: elementwise sigmoid of the logits, with nothing forcing a
    # row to sum to 1 the way a softmax would.
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    np.testing.assert_allclose(probs, expected, atol=1e-6)


@pytest.mark.unit
def test_predict_probs_defaults_to_softmax_for_a_module_without_the_attribute():
    # The contract is "any nn.Module", so a plain module must still work.
    class _Plain(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 2))

    probs = predict_probs(_Plain(), np.zeros((3, 1, 8), dtype=np.float32))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)


@pytest.mark.unit
def test_an_unknown_output_activation_raises_at_construction():
    # Early, because a typo would otherwise silently change every probability
    # the model ever reports rather than failing.
    with pytest.raises(ValueError, match="output_activation must be one of"):
        FCN(in_channels=1, num_classes=2, output_activation="relu")


@pytest.mark.unit
def test_output_activation_survives_a_checkpoint_round_trip(tmp_path):
    # The whole reason it lives on the constructor: a reloaded multi-label
    # checkpoint must still report independent per-label probabilities.
    config = {
        "model": "FCN",
        "init_params": {
            "in_channels": 1,
            "num_classes": 3,
            "filters": [4, 8, 4],
            "output_activation": "sigmoid",
        },
    }
    model = load_model(config, device="cpu", save_path=tmp_path)
    model.save_model(tmp_path / "weights.pt")

    written = json.loads((tmp_path / "weights_config.json").read_text())
    assert written["init_params"]["output_activation"] == "sigmoid"

    reloaded = load_model(
        {"model": "FCN", "model_ckpt_path": str(tmp_path / "weights.pt")}, device="cpu"
    )
    assert reloaded.output_activation == "sigmoid"
    probs = predict_probs(reloaded.eval(), np.zeros((2, 1, 32), dtype=np.float32))
    assert not np.allclose(probs.sum(axis=1), 1.0), "a softmax was re-applied"


@pytest.mark.unit
@pytest.mark.parametrize("config", _TINY_MODEL_CONFIGS)
def test_every_built_in_accepts_output_activation(config):
    params = {
        "in_channels": 1,
        "num_classes": 3,
        "output_activation": "sigmoid",
        **config["init_params"],
    }
    model = load_model({"model": config["model"], "init_params": params}, device="cpu")
    assert model.output_activation == "sigmoid"


# ── evaluate_model: the multi-label branch ────────────────────────────────────

# A 2-D target is one column per output, and every step of the single-label path
# assumes exactly one true class per sample. The branch is picked by the
# FORMULATION rather than the output width, which is what keeps a two-label
# problem off the binary path.


def _multi_label_model(n_outputs: int = 3) -> FCN:
    return FCN(in_channels=1, num_classes=n_outputs, output_activation="sigmoid")


@pytest.mark.unit
def test_a_two_label_problem_does_not_take_the_binary_path(tmp_path):
    # The regression guard on `is_binary = n_model_classes == 2`: with two
    # outputs the width alone is indistinguishable from a binary problem, and
    # the binary path would report pos/neg as a class balance that does not
    # exist here, since a sample can carry both labels or neither.
    data = np.random.default_rng(0).standard_normal((12, 1, 32)).astype("float32")
    rows = np.array([[1, 1], [0, 0], [1, 0], [0, 1]] * 3)

    results = _multi_label_model(2).evaluate_model(
        data, rows, batchsize=4, save_path=tmp_path
    )

    assert not {"pos", "neg"} & set(results), "took the binary branch"
    assert {"subset_accuracy", "bit_accuracy", "hamming_loss"} <= set(results)


@pytest.mark.unit
def test_the_summary_never_calls_anything_just_accuracy(tmp_path):
    # Deliberate: "accuracy" meaning subset accuracy here and exact-match
    # accuracy there is how a cross-formulation comparison goes wrong without
    # anything looking wrong.
    data = np.random.default_rng(1).standard_normal((10, 1, 32)).astype("float32")
    rows = np.array([[1, 0, 1], [0, 1, 0]] * 5)

    results = _multi_label_model().evaluate_model(
        data, rows, batchsize=5, save_path=tmp_path
    )

    assert "accuracy" not in results
    assert results["outputs"] == 3


@pytest.mark.unit
def test_subset_and_bit_accuracy_are_what_they_claim(tmp_path):
    # Pinned against a model whose outputs are forced, so the arithmetic is
    # checkable rather than whatever the random init happens to produce.
    class _Fixed(ModelBase):
        """Emits a fixed logit pattern: output 0 always on, 1 and 2 always off."""

        def __init__(self):
            super().__init__()
            self._record_init_params(locals())
            self.output_activation = "sigmoid"
            self.unused = nn.Linear(1, 1)

        def forward(self, x):
            logits = torch.full((x.shape[0], 3), -10.0)
            logits[:, 0] = 10.0
            return logits

    model = _Fixed().configure(device="cpu", save_path=tmp_path)
    # Truth: output 0 right every time, output 1 wrong every time, output 2
    # right every time. So bit accuracy = 2/3 and subset accuracy = 0.
    rows = np.array([[1, 1, 0]] * 6)
    data = np.zeros((6, 1, 32), dtype=np.float32)

    results = model.evaluate_model(data, rows, batchsize=3, save_path=tmp_path)

    assert results["subset_accuracy"] == f"{0:.4f}"
    assert results["bit_accuracy"] == f"{2 / 3:.4f}"
    assert results["hamming_loss"] == f"{1 / 3:.4f}"


@pytest.mark.unit
def test_the_per_output_csv_has_one_row_per_output_with_its_own_counts(tmp_path):
    data = np.random.default_rng(2).standard_normal((16, 1, 32)).astype("float32")
    rows = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]] * 4)

    _multi_label_model().evaluate_model(
        data, rows, batchsize=4, save_path=tmp_path, class_names=["7Hz", "19Hz", "37Hz"]
    )

    csv = next(tmp_path.glob("*_test_per_class.csv"))
    table = csv.read_text().splitlines()
    assert len(table) == 4, "a header plus one row per output"
    assert table[0].split(",")[:2] == ["class", "name"]
    assert "7Hz" in table[1] and "37Hz" in table[3]
    # samples is that output's positive count, not a share of the sample total.
    header = table[0].split(",")
    samples = header.index("samples")
    assert [row.split(",")[samples] for row in table[1:]] == ["8", "8", "8"]


@pytest.mark.unit
def test_one_confusion_matrix_per_output_named_after_its_class(tmp_path):
    # A K-by-K matrix over output columns would be meaningless, since its rows
    # could not sum to the sample count. One 2x2 per output instead, each in its
    # own file so a figure can be pointed at without cropping a grid.
    data = np.random.default_rng(3).standard_normal((8, 1, 32)).astype("float32")
    rows = np.array([[1, 0, 1], [0, 1, 0]] * 4)

    _multi_label_model().evaluate_model(
        data, rows, batchsize=4, save_path=tmp_path, class_names=["7Hz", "19Hz", "37Hz"]
    )

    written = sorted(p.name for p in tmp_path.glob("*_confusion_matrix_*.png"))
    assert len(written) == 3
    assert all(any(name in f for f in written) for name in ("7Hz", "19Hz", "37Hz"))
    # No single combined figure is left behind under the old name.
    assert not list(tmp_path.glob("*_confusion_matrix.png"))


@pytest.mark.unit
def test_a_class_name_is_made_safe_for_the_filesystem(tmp_path):
    # Names reach the filesystem and are user-supplied now that an encoder takes
    # a `classes` argument: a slash would write into a subdirectory or fail.
    data = np.random.default_rng(5).standard_normal((6, 1, 32)).astype("float32")
    rows = np.array([[1, 0], [0, 1]] * 3)

    _multi_label_model(2).evaluate_model(
        data, rows, batchsize=3, save_path=tmp_path, class_names=["a/b", "7 Hz"]
    )

    written = sorted(p.name for p in tmp_path.glob("*_confusion_matrix_*.png"))
    assert len(written) == 2, "a slash must not redirect the file elsewhere"
    assert any("a_b" in f for f in written)
    assert any("7_Hz" in f for f in written)


@pytest.mark.unit
def test_two_names_that_slug_alike_do_not_overwrite_each_other(tmp_path):
    # The index in the filename is what keeps one figure from replacing the
    # other when the slugs collide.
    data = np.random.default_rng(6).standard_normal((6, 1, 32)).astype("float32")
    rows = np.array([[1, 0], [0, 1]] * 3)

    _multi_label_model(2).evaluate_model(
        data, rows, batchsize=3, save_path=tmp_path, class_names=["7 Hz", "7/Hz"]
    )

    assert len(list(tmp_path.glob("*_confusion_matrix_*.png"))) == 2


@pytest.mark.unit
def test_the_counts_come_from_sklearn_and_match_the_matrices(tmp_path):
    # The per-output table and the figures have to describe the same thing, so
    # both read the one (n_outputs, 2, 2) array sklearn returns.
    from sklearn.metrics import multilabel_confusion_matrix

    data = np.random.default_rng(7).standard_normal((20, 1, 32)).astype("float32")
    rows = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]] * 5)

    model = _multi_label_model()
    model.evaluate_model(data, rows, batchsize=5, save_path=tmp_path)

    probs = predict_probs(model.eval(), data)
    expected = multilabel_confusion_matrix(rows, (probs >= 0.5).astype(int))

    table = (tmp_path / next(tmp_path.glob("*_test_per_class.csv")).name).read_text()
    header, *body = table.splitlines()
    cols = header.split(",")
    for j, line in enumerate(body):
        cell = dict(zip(cols, line.split(","), strict=True))
        (tn, fp), (fn, tp) = expected[j]
        assert (int(cell["tp"]), int(cell["tn"]), int(cell["fp"]), int(cell["fn"])) == (
            tp,
            tn,
            fp,
            fn,
        )
        assert int(cell["samples"]) == tp + fn


@pytest.mark.unit
def test_an_output_with_one_class_present_keeps_a_nan_auc(tmp_path):
    # Output 2 is never present, so it has nothing to separate. It must not drag
    # the macro AUC to nan, and it must not be silently dropped from the table.
    data = np.random.default_rng(4).standard_normal((12, 1, 32)).astype("float32")
    rows = np.array([[1, 0, 0], [0, 1, 0]] * 6)

    results = _multi_label_model().evaluate_model(
        data, rows, batchsize=4, save_path=tmp_path
    )

    assert results["auc"] != "nan"
    # And it says it is an average over two of the three, the same way a class
    # absent from a single-label test split labels its macro AUC partial.
    assert "partial, 2/3 outputs" in results["auc"]
    csv = next(tmp_path.glob("*_test_per_class.csv")).read_text().splitlines()
    assert csv[3].split(",")[-1] == "nan", "the empty output reports no AUC"
