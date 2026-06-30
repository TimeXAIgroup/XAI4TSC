"""Unit tests for model persistence/registry — light torch use, no training."""

import pytest
import torch
from torch import nn

from xai4tsc.models.base import ModelBase
from xai4tsc.models.models import (
    LSTM,
    MODELS,
    XLSTM,
    LeNet,
    PatchTST,
    ResNet,
    _parallel_mlstm,
    _ResidualBlock,
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
    assert (tmp_path / "config.json").exists()


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


@pytest.mark.unit
def test_load_model_unknown_name_raises():
    with pytest.raises(ValueError):
        load_model({"model": "no_such_model"})


# ── save → load round trip ────────────────────────────────────────────────────


@pytest.mark.unit
def test_save_then_load_round_trip_preserves_weights(tmp_path):
    original = load_model(_TINY_FCN, device="cpu", save_path=tmp_path)
    # config.json (= ckpt) must carry the init_params so the reloaded model is
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
