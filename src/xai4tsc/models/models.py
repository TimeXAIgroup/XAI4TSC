"""Built-in models (``FCN``, ...), ``MODELS`` registry, ``load_model()``."""

import json
import logging
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch ecosystem convention
from torch import nn

from ..models.base import ModelBase
from ..utils.utils import dict_to_args, load_class_from_path, merge_dicts

logger = logging.getLogger(__name__)


def save_model(
    model: nn.Module,
    ckpt: dict | None = None,
    optim_state_dict: dict | None = None,
    save_path: Path | str | None = None,
) -> None:
    """
    Save model weights, optional checkpoint information, and optimizer state.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    ckpt : dict, optional
        Optional additional checkpoint information (epoch, loss, etc.),
        by default None
    optim_state_dict : dict, optional
        Optional optimizer state to save, by default None
    save_path : Path or str, optional
        Path to save to, by default None

    Raises
    ------
    ValueError
        If no path is supplied.
    """
    if save_path is None:
        raise ValueError("No path supplied for saving the model!")
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    # Save the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        },
        save_path / "model_dict.pt",
    )
    logger.info("Saved the models state dict to %s", save_path / "model_dict.pt")

    # Save the optimizer state dict if supplied
    if optim_state_dict is not None:
        torch.save({"optim": optim_state_dict}, save_path / "optim_dict.pt")
        logger.info("Saved the optimizer state dict to %s", save_path / "optim_dict.pt")

    # Save checkpoint information, named after the checkpoint it describes so the
    # pairing is unambiguous (model_dict.pt -> model_dict_config.json).
    if ckpt is None:
        logger.warning("Saved a model dictionary without checkpoint information!")
    cfg_path = ModelBase.config_path_for(save_path / "model_dict.pt")
    with open(cfg_path, "w") as f:
        json.dump(ckpt, f)
    logger.info("Saved the model configuration to %s", cfg_path)


def load_ext_model(
    model_name: str,
    class_name: str,
    model_path: Path | str,
    model_params: dict | None = None,
) -> ModelBase:
    """
    Load an external model class and instantiate a model.

    Executes arbitrary Python from *model_path* via
    :func:`~xai4tsc.utils.utils.load_class_from_path`; the experiment runner
    gates this behind ``general.allow_external_code``.

    Parameters
    ----------
    model_name : str
        Name of the model.
    class_name : str
        Model class name to retrieve from the source file.
    model_path : Path | str
        Path to the source file containing the class definition.
    model_params : dict, optional
        Parameters to instantiate the model, by default None

    Returns
    -------
    ModelBase
        Instantiated model with operational attributes set.
    """
    user_model_class = load_class_from_path(model_path, class_name)
    filtered_params = (
        dict_to_args(model_params, user_model_class) if model_params else {}
    )
    model = user_model_class(**(filtered_params or {}))
    model.name = model_name
    model._init_params = filtered_params
    return model


def _set_model_attrs(
    model: ModelBase,
    name: str,
    device: str | torch.device,
    save_path: str | Path | None,
    init_params: dict,
) -> None:
    """Set operational attributes on a freshly constructed ModelBase instance."""
    model.name = name
    model._init_params = init_params or {}
    model.device = torch.device(device) if isinstance(device, str) else device
    model.save_path = Path(save_path) if save_path is not None else None
    model.to(model.device)


def _drop_none(cfg: dict) -> dict:
    """Return *cfg* without keys whose value is ``None``, recursively."""
    cleaned = {}
    for key, value in cfg.items():
        if value is None:
            continue
        cleaned[key] = _drop_none(value) if isinstance(value, dict) else value
    return cleaned


def _read_checkpoint_config(checkpoint_path: Path) -> tuple[dict, Path | None]:
    """
    Read the config file belonging to *checkpoint_path*.

    Prefers ``<stem>_config.json`` beside the checkpoint and falls back to a
    directory-level ``config.json``, the name earlier versions wrote.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the ``.pt`` checkpoint.

    Returns
    -------
    tuple[dict, Path or None]
        The config mapping with ``None`` values dropped, and the file it came
        from, or ``({}, None)`` when no config file exists.
    """
    candidates = [
        ModelBase.config_path_for(checkpoint_path),
        checkpoint_path.with_name("config.json"),  # legacy, pre-per-checkpoint
    ]
    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            with open(cfg_path) as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable config %s: %s", cfg_path, exc)
            continue
        if isinstance(loaded, dict):
            return _drop_none(loaded), cfg_path
        logger.warning("Ignoring %s: expected a JSON object.", cfg_path)
    return {}, None


def _log_config_merge(loaded_cfg: dict, cfg_path: Path, model_config: dict) -> None:
    """
    Report what a checkpoint's config file contributed, and what it overrode.

    The file takes precedence over the supplied config, so a silent merge would
    let the architecture actually used differ from the one the user is reading.

    Parameters
    ----------
    loaded_cfg : dict
        Config read from *cfg_path*, already stripped of ``None`` values.
    cfg_path : Path
        File the config was read from.
    model_config : dict
        Config supplied by the caller, which *loaded_cfg* overrides.
    """
    supplied = []
    for key, value in loaded_cfg.items():
        if isinstance(value, dict) and value:
            supplied.append(f"{key} ({', '.join(value)})")
        else:
            supplied.append(f"{key}={value}")
    logger.info(
        "%s supplied: %s; these take precedence.", cfg_path.name, ", ".join(supplied)
    )
    logger.debug("Full config read from %s: %s", cfg_path, loaded_cfg)

    # Flag only genuine conflicts: a value the caller set that the file changes.
    for key, value in loaded_cfg.items():
        if key not in model_config:
            continue
        given = model_config[key]
        if isinstance(value, dict) and isinstance(given, dict):
            for sub, sub_value in value.items():
                if sub in given and given[sub] is not None and given[sub] != sub_value:
                    logger.warning(
                        "%s overrides configured %s.%s: %s -> %s (the weights only "
                        "fit the architecture they were trained with).",
                        cfg_path.name,
                        key,
                        sub,
                        given[sub],
                        sub_value,
                    )
        elif given is not None and given != value:
            logger.warning(
                "%s overrides configured %s: %s -> %s (the weights only fit the "
                "architecture they were trained with).",
                cfg_path.name,
                key,
                given,
                value,
            )


def _epoch_from_filename(checkpoint_path: Path) -> int | None:
    """Return the epoch encoded in a ``<Model>_epoch_<N>.pt`` name, else ``None``."""
    match = re.search(r"_epoch_(\d+)$", checkpoint_path.stem)
    return int(match.group(1)) if match else None


def _build_from_class(
    model_class: type,
    state_dict: dict,
    model_params: dict | None,
    name: str,
    device: str | torch.device,
    save_path: str | Path | None,
) -> ModelBase:
    """Instantiate *model_class*, load *state_dict* into it, and set attributes."""
    filtered_params = dict_to_args(model_params, model_class) if model_params else {}
    model = model_class(**(filtered_params or {}))
    model.load_state_dict(state_dict)
    _set_model_attrs(model, name, device, save_path, filtered_params)
    return model


def load_model(
    model_config: dict, device: str = "cpu", save_path: str | Path | None = None
) -> ModelBase:
    """
    Load or instantiate a model from a config dict.

    Looks for a ``.pt`` checkpoint path or a Python class path in
    *model_config* and returns a ready-to-use :class:`ModelBase` instance.

    A ``.pt`` checkpoint may be either the self-describing dict written by
    :func:`save_model` (carrying ``model_class``) or a bare state dict as
    written by :meth:`~xai4tsc.models.base.ModelBase.save_model` - the form of
    every ``<Model>_epoch_<N>.pt`` the experiment runner produces.  The class to
    build is resolved in order from the checkpoint's own ``model_class``, the
    ``model`` registry name, and finally ``class_name`` / ``class_path``.  A
    ``config.json`` beside the checkpoint is merged in and takes precedence, so
    the architecture the weights belong to need not be restated in
    *model_config*.

    Parameters
    ----------
    model_config : dict
        Model configuration dict.  Recognised keys: ``model`` (registry name),
        ``model_ckpt_path`` (.pt checkpoint), ``class_name`` / ``class_path``
        (dynamic class loading from an external source file), ``init_params``.
    device : str, optional
        Device on which to load the model, by default ``"cpu"``.
    save_path : str or Path, optional
        Directory used for saving checkpoints during training.  Passed through
        to the model instance as ``model.save_path``.

    Returns
    -------
    ModelBase
        The model instance with ``name``, ``device``, ``save_path``, and
        ``_init_params`` set.

    Raises
    ------
    FileNotFoundError
        If ``model_ckpt_path`` names a ``.pt`` file that does not exist.
    ValueError
        If no class can be resolved for a supplied checkpoint, or if neither a
        checkpoint path nor a registered ``model`` name is given.
    """
    model = None
    if "model_ckpt_path" in model_config:
        # A checkpoint is supplied
        path = Path(model_config["model_ckpt_path"]).absolute()
        format = path.suffix
        if format == ".h5":
            # TODO implement
            logger.info("Model datatype not implemented, please use .pt")
        elif format == ".pt":
            if not path.is_file():
                siblings = (
                    sorted(f.name for f in path.parent.glob("*.pt"))
                    if path.parent.is_dir()
                    else []
                )
                hint = (
                    f" Checkpoints present in {path.parent}: {siblings}."
                    if siblings
                    else f" No '.pt' files found in {path.parent}."
                )
                raise FileNotFoundError(
                    f"Model checkpoint '{path}' does not exist.{hint}"
                )

            # Read the checkpoint's own config and merge it in. It describes the
            # architecture the weights belong to, so it takes precedence.
            loaded_cfg, cfg_path = _read_checkpoint_config(path)
            if cfg_path is not None:
                _log_config_merge(loaded_cfg, cfg_path, model_config)
                model_config = merge_dicts(loaded_cfg, model_config)
            model_params = model_config.get("init_params")

            ckpt_epoch = loaded_cfg.get("epoch")
            if ckpt_epoch is None:
                ckpt_epoch = _epoch_from_filename(path)
            if ckpt_epoch is None:
                logger.warning(
                    "Cannot tell how many epochs '%s' was trained for (no 'epoch' "
                    "in a %s and the name is not '<Model>_epoch_<N>.pt'). Epoch "
                    "numbering restarts at 1 and will not reflect total training.",
                    path.name,
                    ModelBase.config_path_for(path).name,
                )

            # Load model checkpoint. Either the self-describing dict written by
            # save_model(), or a bare state dict as written by
            # ModelBase.save_model() - the form every <Model>_epoch_<N>.pt the
            # experiment runner produces takes.
            model_ckpt = torch.load(path, map_location=device, weights_only=False)
            wrapped = isinstance(model_ckpt, dict) and "model_state_dict" in model_ckpt
            state_dict = model_ckpt["model_state_dict"] if wrapped else model_ckpt
            ckpt_class = (
                model_ckpt.get("model_class") if isinstance(model_ckpt, dict) else None
            )

            origin = None

            # 1) The checkpoint names its own class.
            if ckpt_class is not None:
                model_class = MODELS.get(str(ckpt_class).lower())
                if model_class is not None:
                    model = _build_from_class(
                        model_class,
                        state_dict,
                        model_params,
                        model_config["model"],
                        device,
                        save_path,
                    )
                    origin = "class from the checkpoint's 'model_class'"

            # 2) A bare state dict: fall back to the registry name in the config.
            registry_name = str(model_config.get("model", "")).lower()
            if model is None and registry_name in MODELS:
                model = _build_from_class(
                    MODELS[registry_name],
                    state_dict,
                    model_params,
                    model_config["model"],
                    device,
                    save_path,
                )
                origin = "class from the MODELS registry"

            # 3) No registered model class found - try loading from source
            if model is None and "class_path" in model_config:
                model = load_ext_model(
                    model_name=model_config["model"],
                    class_name=model_config["class_name"],
                    model_path=model_config["class_path"],
                    model_params=model_params,
                )
                model.load_state_dict(state_dict)
                _set_model_attrs(model, model_config["model"], device, save_path, {})
                origin = f"class from {model_config['class_path']}"

            if model is not None:
                model.best_epoch = ckpt_epoch
                model._loaded_from = path
                logger.info(
                    "Loaded %s from checkpoint %s (%s; %s)",
                    model.name,
                    path,
                    origin,
                    f"trained {ckpt_epoch} epoch(s)"
                    if ckpt_epoch is not None
                    else "training history unknown",
                )

            if model is None:
                tried = (
                    f"its 'model_class' ({ckpt_class!r}) is not in the MODELS registry"
                    if ckpt_class is not None
                    else "it carries no 'model_class' key"
                )
                raise ValueError(
                    f"Unable to instantiate a model for checkpoint '{path}': "
                    f"{tried}, the configured model name "
                    f"{model_config.get('model')!r} is not registered either, "
                    "and no 'class_name'/'class_path' was supplied. "
                    f"Registered models: {sorted(MODELS)}."
                )
        elif format == ".ckpt":
            # TODO implement
            logger.info("Model datatype not implemented, please use .pt")
    elif model_config["model"].lower() in MODELS:
        model_class = MODELS[model_config["model"].lower()]
        model_params = model_config.get("init_params")
        filtered_params = (
            dict_to_args(model_params, model_class) if model_params else {}
        )
        model = model_class(**(filtered_params or {}))
        _set_model_attrs(
            model, model_config["model"], device, save_path, filtered_params
        )
        logger.info(
            "Built %s from the MODELS registry (no checkpoint supplied).", model.name
        )
    else:
        raise ValueError("No model checkpoint path supplied!")

    return model


class FCN(ModelBase):
    """
    Fully Convolutional Network for time series classification.

    A dataset-agnostic 3-block FCN (Wang et al., 2017) that works on any UCR/UEA
    dataset regardless of sequence length or number of channels, because the final
    global average pooling collapses the time axis.

    Architecture: three ``[Conv1d → BatchNorm → ReLU]`` blocks followed by
    global average pooling and a linear classifier.

    Input shape:  ``(B, in_channels, T)``
    Output shape: ``(B, num_classes)`` (logits)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate, >1 for multivariate).
    num_classes : int
        Number of output classes.
    filters : tuple[int, int, int]
        Number of conv filters in each of the three blocks.
        Defaults to ``(128, 256, 128)`` as in the original paper.
    kernel_sizes : tuple[int, int, int]
        Kernel sizes for the three blocks. Defaults to ``(8, 5, 3)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: tuple[int, int, int] = (128, 256, 128),
        kernel_sizes: tuple[int, int, int] = (8, 5, 3),
    ) -> None:
        super().__init__()
        channels = [in_channels, *filters]
        blocks = []
        for i in range(3):
            k = kernel_sizes[i]
            blocks += [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(filters[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class LeNet(ModelBase):
    """
    Shallow LeNet-inspired 1D CNN for time series classification.

    Reproduces the shallow model of Wagner et al. (2024) for ECG
    classification: three ``Conv1d`` layers (kernel size 5, stride 2, output
    channels 32 → 64 → 128), each interleaved with ``BatchNorm`` → ``ReLU`` →
    pooling (the first two pools ``MaxPool``, the last ``AvgPool``), followed by
    two fully-connected layers interleaved with ``ReLU``.

    Reference
    ---------
    Wagner et al., *Explaining deep learning for ECG analysis: Building blocks
    for auditing and knowledge discovery*, Computers in Biology and Medicine,
    2024. (LeNet architecture: LeCun et al., 1998.)

    Notes
    -----
    To stay dataset-agnostic (xai4tsc runs on any UCR/UEA length), the final
    ``AvgPool`` is a global :class:`~torch.nn.AdaptiveAvgPool1d` so the
    fully-connected head sees a fixed-size input regardless of ``T``; the
    original fixed-length PTB-XL setup (1000 timesteps) used a plain ``AvgPool``.
    Convolutions use ``padding = kernel_size // 2``.

    Input shape:  ``(B, in_channels, T)``
    Output shape: ``(B, num_classes)`` (logits)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate, >1 for multivariate).
    num_classes : int
        Number of output classes.
    head_hidden : int
        Width of the hidden fully-connected layer. Defaults to 128.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        channels = [in_channels, 32, 64, 128]
        pools = [nn.MaxPool1d(2), nn.MaxPool1d(2), nn.AdaptiveAvgPool1d(1)]
        blocks = []
        for i in range(3):
            blocks += [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
                pools[i],
            ]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        x = self.features(x)
        return self.classifier(x)


class _ResidualBlock(nn.Module):
    """
    One Wang et al. (2017) residual block: three Conv-BN-ReLU layers + shortcut.

    Each convolution uses ``padding="same"`` so the time axis is preserved and
    the residual shortcut can be added without resampling. The final ReLU is
    applied *after* the shortcut is summed in. The shortcut is a ``1x1`` conv +
    BatchNorm when the channel count changes, and a plain BatchNorm otherwise
    (matching the original architecture).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, int, int] = (8, 5, 3),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for k in kernel_sizes:
            layers += [
                nn.Conv1d(ch, out_channels, kernel_size=k, padding="same", bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ]
            ch = out_channels
        # Drop the trailing ReLU — it is applied after the residual sum.
        self.conv_block = nn.Sequential(*layers[:-1])

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        return self.act(self.conv_block(x) + self.shortcut(x))


class ResNet(ModelBase):
    """
    Residual Network for time series classification (Wang et al., 2017).

    The strong TSC baseline from *Time Series Classification from Scratch with
    Deep Neural Networks* — three residual blocks (channel widths
    ``[F, 2F, 2F]`` for feature-map width ``F``), each with three
    ``[Conv1d → BatchNorm → ReLU]`` layers (kernel sizes ``8, 5, 3``) and a
    residual shortcut, followed by global average pooling and a linear
    classifier. It is the residual sibling of :class:`FCN` from the same paper.

    Dataset-agnostic: ``padding="same"`` keeps the time axis intact through the
    blocks and the global average pool collapses it, so the model runs on any
    UCR/UEA length ``T`` and channel count.

    Reference
    ---------
    Z. Wang, W. Yan, and T. Oates, *Time Series Classification from Scratch with
    Deep Neural Networks: A Strong Baseline*, IJCNN, 2017.

    Input shape:  ``(B, in_channels, T)``
    Output shape: ``(B, num_classes)`` (logits)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate, >1 for multivariate).
    num_classes : int
        Number of output classes.
    n_feature_maps : int
        Base number of convolution filters ``F``; blocks use ``F, 2F, 2F``.
        Defaults to 64, as in the original paper.
    kernel_sizes : tuple[int, int, int]
        Kernel sizes of the three convolutions inside each residual block.
        Defaults to ``(8, 5, 3)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_feature_maps: int = 64,
        kernel_sizes: tuple[int, int, int] = (8, 5, 3),
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _ResidualBlock(in_channels, n_feature_maps, kernel_sizes),
            _ResidualBlock(n_feature_maps, n_feature_maps * 2, kernel_sizes),
            _ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, kernel_sizes),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class LSTM(ModelBase):
    """
    Recurrent LSTM (bidirectional by default) for time series classification.

    A stack of LSTM layers reads the series along the time axis; the per-timestep
    outputs are mean-pooled over time and passed to a linear classification head.
    Channels are the LSTM input features, so the model handles univariate and
    multivariate series alike.

    This is the dependable recurrent baseline. The state-of-the-art recurrent
    upgrade is :class:`XLSTM` in this module.

    Notes
    -----
    Mean-pooling over time (rather than taking the last hidden state) keeps the
    model agnostic to series length ``T`` and gives smoother gradients for
    attribution.

    ``dropout`` only takes effect between stacked layers and is therefore ignored
    when ``num_layers == 1`` (PyTorch ``nn.LSTM`` semantics).

    **On xLSTM and the dependency tradeoff.** :class:`XLSTM` is the modern,
    state-of-the-art successor to this model (Beck et al., 2024). The official
    NX-AI ``xlstm`` package ships the full sLSTM + mLSTM stack, but its fast path
    relies on **Triton/CUDA kernels** (plus extra dependencies such as ``einops``,
    ``dacite`` and a ``ninja`` build toolchain). That would break the CPU-only
    ``ubuntu-latest`` CI and the M-series/MPS dev loop, and a compiled kernel is
    hostile to Captum's gradient attribution. xai4tsc therefore does **not** depend
    on it: :class:`XLSTM` clean-room reimplements only the *parallelizable* mLSTM
    block in pure PyTorch (no kernel) and intentionally omits the strictly
    recurrent sLSTM block — the part whose value is the fast kernel we are avoiding.

    Captum note: cuDNN's RNN backward does not support the double-backward some
    Captum methods need, but only on **CUDA + cuDNN**. CPU CI and MPS are
    unaffected; on a CUDA GPU, run attribution with ``cudnn`` disabled.

    Input shape:  ``(B, in_channels, T)``
    Output shape: ``(B, num_classes)`` (logits)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate, >1 for multivariate).
    num_classes : int
        Number of output classes.
    hidden_size : int
        Number of features in each LSTM hidden state. Defaults to 128.
    num_layers : int
        Number of stacked LSTM layers. Defaults to 2.
    bidirectional : bool
        If ``True`` (default), use a bidirectional LSTM; the head input width is
        then ``2 * hidden_size``.
    dropout : float
        Dropout applied between stacked LSTM layers (ignored when
        ``num_layers == 1``). Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_in = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        # (B, C, T) -> (B, T, C); C is the per-timestep LSTM input feature dim.
        out, _ = self.lstm(x.transpose(1, 2))  # (B, T, H * num_directions)
        feat = out.mean(dim=1)  # length-agnostic temporal pooling
        return self.classifier(feat)


class PatchTST(ModelBase):
    """
    Channel-independent patch transformer for time series classification.

    Adapts PatchTST (Nie et al., 2023) to classification. Each channel is
    processed independently: the series is split into (optionally overlapping)
    patches, each patch is linearly embedded to ``d_model``, a sinusoidal
    positional encoding is added, and a stack of standard transformer-encoder
    layers attends across patches. Patch representations are mean-pooled over
    patches and then over channels, giving a fixed-size feature regardless of the
    channel count ``C`` or series length ``T``, which a linear head maps to logits.

    Reference
    ---------
    Nie, Nguyen, Sinthong & Kalagnanam, *A Time Series is Worth 64 Words:
    Long-term Forecasting with Transformers*, ICLR 2023 (PatchTST). For a
    transformer purpose-built for multivariate TSC, see ConvTran (Foumani et al.,
    2024) — a documented future alternative, not implemented here.

    Notes
    -----
    A **sinusoidal** positional encoding (computed per forward pass) is used
    rather than a learnable one so the model is agnostic to the number of patches,
    and therefore to ``T`` — the constructor is not told the series length.
    Series shorter than ``patch_len`` are right-padded (edge replication) up to a
    single patch; longer series are right-padded so the patches tile ``T`` exactly.

    Captum note: the internal patching does not block attribution — gradients flow
    through the patch-embedding ``Linear`` back to the raw ``(B, C, T)`` input.

    Input shape:  ``(B, in_channels, T)``
    Output shape: ``(B, num_classes)`` (logits)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate, >1 for multivariate).
    num_classes : int
        Number of output classes.
    d_model : int
        Transformer embedding dimension. Must be divisible by ``nhead``.
        Defaults to 128.
    patch_len : int
        Length of each patch. Defaults to 16.
    stride : int
        Step between consecutive patches (``< patch_len`` gives overlap).
        Defaults to 8.
    num_layers : int
        Number of transformer-encoder layers. Defaults to 3.
    nhead : int
        Number of attention heads. Must divide ``d_model``. Defaults to 8.
    dim_feedforward : int
        Width of the encoder feed-forward sublayer. Defaults to 256.
    dropout : float
        Dropout used inside the transformer encoder. Defaults to 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            msg = f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            raise ValueError(msg)
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _positional_encoding(
        n: int, d: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a standard sinusoidal positional encoding of shape ``(1, n, d)``."""
        pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=dtype) * (-math.log(1e4) / d)
        )
        pe = torch.zeros(n, d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        b, c, t = x.shape
        # Channel independence: each channel becomes its own sequence.
        xi = x.reshape(b * c, 1, t)
        # Right-pad (edge replication) so the patches tile the series exactly.
        if t < self.patch_len:
            pad = self.patch_len - t
        else:
            rem = (t - self.patch_len) % self.stride
            pad = (self.stride - rem) % self.stride
        if pad:
            xi = F.pad(xi, (0, pad), mode="replicate")
        xi = xi.squeeze(1)
        patches = xi.unfold(-1, self.patch_len, self.stride)  # (B*C, N, patch_len)
        emb = self.patch_embed(patches)  # (B*C, N, d_model)
        emb = emb + self._positional_encoding(
            emb.shape[1], emb.shape[2], emb.device, emb.dtype
        )
        enc = self.encoder(emb)  # (B*C, N, d_model)
        feat = enc.mean(dim=1).reshape(b, c, -1).mean(dim=1)  # (B, d_model)
        return self.head(feat)


def _parallel_mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Parallel, numerically stabilized mLSTM cell (no time loop).

    Computes the matrix-memory mLSTM (Beck et al., 2024) in its parallel form: an
    attention-like ``(Q Kᵀ) ⊙ D`` interaction where ``D`` is the cumulative
    forget-gate decay matrix and the exponential input gate enters in log space.
    Stabilized by subtracting the row-wise max of the log-decay matrix. Equivalent
    to the sequential recurrence but fully parallel, so it needs no custom kernel.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query/key/value, shape ``(B, num_heads, S, head_dim)``.
    igate_preact, fgate_preact : torch.Tensor
        Input- and forget-gate pre-activations, shape ``(B, num_heads, S, 1)``.
    eps : float
        Numerical floor added to the normalizer.

    Returns
    -------
    torch.Tensor
        Cell output ``(B, num_heads, S, head_dim)`` (before the output gate).
    """
    b, nh, s, dh = q.shape
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    cumsum = torch.cat(
        [
            torch.zeros(b, nh, 1, 1, device=q.device, dtype=q.dtype),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    rep = cumsum.expand(b, nh, s + 1, s + 1)
    # entry [t, s'] = sum of log-forget gates over (s', t]; crop the leading 0 row/col
    log_fg = (rep - rep.transpose(-2, -1))[:, :, 1:, 1:]
    ltr = torch.tril(torch.ones(s, s, dtype=torch.bool, device=q.device))
    log_fg = log_fg.masked_fill(~ltr, float("-inf"))
    log_d = log_fg + igate_preact.transpose(-2, -1)  # add log input gate per column
    max_log_d = log_d.max(dim=-1, keepdim=True).values
    d = torch.exp(log_d - max_log_d)
    qk = q @ (k / math.sqrt(dh)).transpose(-2, -1)
    c = qk * d
    normalizer = torch.maximum(c.sum(-1, keepdim=True).abs(), torch.exp(-max_log_d))
    return (c / (normalizer + eps)) @ v


class _MLSTMBlock(nn.Module):
    """Pre-norm mLSTM block: matrix-memory cell + output gate, then a feed-forward."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.igate = nn.Linear(embed_dim, num_heads)
        self.fgate = nn.Linear(embed_dim, num_heads)
        self.ogate = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, S, E)`` to multi-head ``(B, num_heads, S, head_dim)``."""
        b, s, _ = t.shape
        return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, S, embed_dim)``."""
        h = self.norm1(x)
        b, s, e = h.shape
        q, k, v = self._heads(self.q(h)), self._heads(self.k(h)), self._heads(self.v(h))
        ig = self.igate(h).transpose(1, 2).unsqueeze(-1)  # (B, num_heads, S, 1)
        fg = self.fgate(h).transpose(1, 2).unsqueeze(-1)
        cell = _parallel_mlstm(q, k, v, ig, fg)  # (B, num_heads, S, head_dim)
        cell = cell.transpose(1, 2).reshape(b, s, e)
        gated = torch.sigmoid(self.ogate(h)) * cell
        x = x + self.drop(self.proj(gated))
        return x + self.drop(self.ffn(self.norm2(x)))


class XLSTM(ModelBase):
    """
    xLSTM (mLSTM) for time series classification — clean-room, pure PyTorch.

    State-of-the-art recurrent model: a stack of pre-norm **mLSTM** blocks
    (matrix-memory LSTM with exponential gating, Beck et al., 2024) reads the
    series, and the per-timestep representations are mean-pooled over time and
    mapped to logits by a linear head. Channels are first projected to
    ``embed_dim`` per timestep, so univariate and multivariate series are handled
    alike. The dependable, lighter recurrent baseline is :class:`LSTM`.

    Reference
    ---------
    Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer, Brandstetter &
    Hochreiter, *xLSTM: Extended Long Short-Term Memory*, NeurIPS 2024.

    Notes
    -----
    **Dependency tradeoff (see also** :class:`LSTM` **).** The official NX-AI
    ``xlstm`` package ships the full sLSTM + mLSTM stack but its fast path relies
    on **Triton/CUDA kernels** (plus ``einops``, ``dacite`` and a ``ninja`` build
    toolchain), which would break the CPU-only CI and the MPS dev loop and is
    hostile to Captum's gradient attribution. xai4tsc therefore depends on none of
    it: this model clean-room reimplements only the **mLSTM** block in its
    *parallel* form (:func:`_parallel_mlstm`) — fully differentiable, no kernel —
    and **intentionally omits the sLSTM block** (the strictly recurrent, scalar-
    memory part whose value is precisely the fast kernel being avoided).

    Mean-pooling over time keeps the head independent of series length ``T``.

    Input shape:  ``(B, in_channels, T)``
    Output shape: ``(B, num_classes)`` (logits)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for univariate, >1 for multivariate).
    num_classes : int
        Number of output classes.
    embed_dim : int
        Model width. Must be divisible by ``num_heads``. Defaults to 128.
    num_blocks : int
        Number of stacked mLSTM blocks. Defaults to 2.
    num_heads : int
        Number of mLSTM heads. Must divide ``embed_dim``. Defaults to 4.
    dropout : float
        Dropout applied to the block residual branches. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = (
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
            raise ValueError(msg)
        self.embed = nn.Linear(in_channels, embed_dim)
        self.blocks = nn.ModuleList(
            [_MLSTMBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input of shape ``(B, C, T)``."""
        h = self.embed(x.transpose(1, 2))  # (B, T, embed_dim)
        for block in self.blocks:
            h = block(h)
        feat = self.norm(h).mean(dim=1)  # length-agnostic temporal pooling
        return self.head(feat)


MODELS = {
    "fcn": FCN,
    "lenet": LeNet,
    "resnet": ResNet,
    "lstm": LSTM,
    "patchtst": PatchTST,
    "xlstm": XLSTM,
}


def register_model(name: str, model_class: type) -> None:
    """
    Register a custom model class.

    The class must be a :class:`~xai4tsc.models.base.ModelBase` subclass — the
    runner relies on the training, prediction, and evaluation methods provided
    there, so a plain :class:`torch.nn.Module` is not sufficient. After
    registration it is available by *name* in experiment configs.

    Parameters
    ----------
    name:
        Key used to look up the model (matched case-insensitively in configs).
    model_class:
        A :class:`~xai4tsc.models.base.ModelBase` subclass.

    Raises
    ------
    TypeError
        If *model_class* is not a :class:`~xai4tsc.models.base.ModelBase`
        subclass.
    """
    if not (isinstance(model_class, type) and issubclass(model_class, ModelBase)):
        raise TypeError(
            f"Model '{name}' must be a ModelBase subclass, got {model_class!r}."
        )
    MODELS[name] = model_class
