"""
Shared resolution of a config string to the library object it names.

The package is typed against the library types - a :class:`torch.nn.Module` loss, a
:class:`torch.optim.Optimizer` subclass, an sklearn-style target encoder - so the
only gap is that YAML cannot hold a Python object. That is a serialization problem
rather than an abstraction, which is why each family is a table plus a dotted-path
escape hatch instead of a registry: there is nothing to curate, and a loss that
does not exist yet still works.

This module owns what the three families share: the type check, the case-folded
table lookup, the retired-name message, the dotted-path import, and the wording of
the error that names the available keys. Each family then constructs in the way
that suits it - an encoder takes no arguments, a loss is built with its parameters,
an optimizer is deferred into a :func:`functools.partial` because it needs
``model.parameters()``, which only exists inside ``train_model``.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger("xai4tsc.runner.lookups")


@dataclass(frozen=True)
class Lookup:
    """
    One config-string-to-object family.

    Parameters
    ----------
    table : dict
        Config name (lower case) -> the class or factory it denotes.
    key : str
        The config key this family is configured by, used in every message so an
        error names the YAML key the user wrote (``"encode"``, ``"loss_func"``).
    label : str
        The thing being named, article included (``"an encoder"``, ``"a loss"``),
        so the sentences read correctly.
    example : str
        A dotted path to show in the "available" line.
    retired : dict, optional
        Names that used to work, mapped to the reason they no longer do. A retired
        name must say what to use instead: a config carrying one stops running.
    """

    table: dict[str, Callable]
    key: str
    label: str
    example: str
    retired: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> Callable:
        """
        Return the class or factory that *name* denotes.

        Table lookups are case-folded, because configs in the wild spell a loss
        ``"CrossEntropy"``; a dotted path is left alone, since
        ``torch.nn.BCEWithLogitsLoss`` cannot resolve lower-cased.

        Parameters
        ----------
        name : str
            A key of :attr:`table`, or a dotted import path to a class.

        Returns
        -------
        Callable
            The class or factory. Constructing it is the caller's business.

        Raises
        ------
        ValueError
            If *name* is not a string, is retired, or resolves to nothing.
            Falling back to a default instead would run a different experiment
            and report plausible numbers for it.
        """
        if not isinstance(name, str):
            raise ValueError(
                f"{self.key} must be a string naming {self.label}, got "
                f"{type(name).__name__} ({name!r}). {self._available()}"
            )
        folded = name.lower()
        if folded in self.table:
            return self.table[folded]
        if folded in self.retired:
            raise ValueError(
                f"{self.key}: '{name}' is no longer available because "
                f"{self.retired[folded]}. {self._available()}"
            )
        if "." in name:
            return self._from_dotted_path(name)
        raise ValueError(f"Unknown {self.key}: '{name}'. {self._available()}")

    def construct(self, name: str, params: dict | None = None) -> object:
        """
        Resolve *name* and build it with *params*.

        Parameters
        ----------
        name : str
            As :meth:`resolve`.
        params : dict, optional
            Constructor keyword arguments.

        Returns
        -------
        object
            The constructed instance.

        Raises
        ------
        ValueError
            If construction fails. The parameters come from YAML, so a wrong name
            or a wrong shape is a config error and is reported as one.
        """
        factory = self.resolve(name)
        params = params or {}
        try:
            return factory(**params)
        except (TypeError, ValueError) as exc:
            if params:
                raise ValueError(
                    f"{self.key}: '{name}' could not be constructed from {params!r} "
                    f"({exc}). Check the parameter names and values against "
                    f"{getattr(factory, '__name__', factory)}."
                ) from exc
            raise ValueError(
                f"{self.key}: '{name}' could not be constructed without arguments "
                f"({exc}). Supply one that takes none, or pass the object directly "
                "from Python instead of via YAML."
            ) from exc

    def _available(self) -> str:
        """Return the sentence listing what this family accepts."""
        return (
            f"Available: {sorted(self.table)}, or a dotted path to {self.label} "
            f"class, e.g. '{self.example}'."
        )

    def _from_dotted_path(self, path: str) -> Callable:
        """
        Import the class or factory named by a dotted *path*.

        Parameters
        ----------
        path : str
            Dotted path, e.g. ``"torch.optim.NAdam"``.

        Returns
        -------
        Callable
            The imported attribute.

        Raises
        ------
        ValueError
            If the module or the attribute does not exist, or it is not callable.
        """
        module_path, _, attr = path.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"Unknown {self.key}: '{path}' - module '{module_path}' could not be "
                f"imported ({exc}). {self._available()}"
            ) from exc
        candidate = getattr(module, attr, None)
        if candidate is None or not callable(candidate):
            raise ValueError(
                f"Unknown {self.key}: '{path}' - '{module_path}' has no callable "
                f"'{attr}'. {self._available()}"
            )
        logger.info(
            "Resolved %s '%s' to %s.",
            self.key,
            path,
            getattr(candidate, "__name__", candidate),
        )
        return candidate
