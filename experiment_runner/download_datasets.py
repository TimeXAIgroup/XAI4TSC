"""
Pre-download UCR/UEA dataset archives into the experiment cache.

``ensure_datasets_cached`` is called before the experiment loop.  It:

1. Checks which datasets are already in the cache (fast no-op if all present).
2. Downloads the full bulk archive zip in one shot when the wildcard is active
   and at least half the requested datasets are missing.
3. Extracts and flattens the zip so each dataset lands at
   ``{cache_dir}/{name}/{name}_TRAIN.ts`` - the path sktime expects via
   its ``extract_path`` parameter.

Any dataset still missing afterwards (no wildcard, or a failed bulk download)
is downloaded by sktime into the same layout when ``UcrUeaDataset`` loads it.
A download error then fails that dataset inside the experiment loop, which
continues with the next one.
"""

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import tqdm
from sktime.datasets.tsc_dataset_names import (
    multivariate as _UEA_NAMES,  # noqa: N812 — uppercase denotes a constant
)
from sktime.datasets.tsc_dataset_names import (
    univariate as _UCR_NAMES,  # noqa: N812 — uppercase denotes a constant
)

logger = logging.getLogger("xai4tsc.runner.download")

_BASE_URL = "https://timeseriesclassification.com/aeon-toolkit"
UCR_BULK_URL = f"{_BASE_URL}/Archives/Univariate2018_ts.zip"
UEA_BULK_URL = f"{_BASE_URL}/Archives/Multivariate2018_ts.zip"

_UCR_SET = set(_UCR_NAMES)
_UEA_SET = set(_UEA_NAMES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_datasets_cached(
    cache_dir: Path,
    dataset_names: list[str],
    uses_ucr_wildcard: bool = False,
    uses_uea_wildcard: bool = False,
) -> None:
    """
    Ensure every dataset in *dataset_names* has its ``.ts`` files cached.

    Parameters
    ----------
    cache_dir:
        Root directory where ``{name}/{name}_TRAIN.ts`` files are stored.
    dataset_names:
        Fully-expanded list of dataset names (after UCR/UEA wildcard resolution).
    uses_ucr_wildcard:
        ``True`` when the original config used ``dataset: "UCR"``.  Triggers
        the bulk archive download; without a wildcard nothing is downloaded
        here and sktime fetches each missing dataset when it is loaded.
    uses_uea_wildcard:
        Same for ``dataset: "UEA"``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [n for n in dataset_names if not _is_cached(cache_dir, n)]
    if not missing:
        logger.info(
            "All %d dataset(s) already cached — skipping download.",
            len(dataset_names),
        )
        return

    logger.info("%d / %d dataset(s) not yet cached.", len(missing), len(dataset_names))

    ucr_missing = {n for n in missing if n in _UCR_SET}
    uea_missing = {n for n in missing if n in _UEA_SET}

    bulk_failed = False

    if uses_ucr_wildcard and len(ucr_missing) >= len(dataset_names) // 2:
        logger.info(
            "UCR wildcard active — downloading bulk archive (%d missing).",
            len(ucr_missing),
        )
        bulk_failed |= not _download_bulk(cache_dir, UCR_BULK_URL)
        ucr_missing = {n for n in ucr_missing if not _is_cached(cache_dir, n)}

    if uses_uea_wildcard and len(uea_missing) >= len(dataset_names) // 2:
        logger.info(
            "UEA wildcard active — downloading bulk archive (%d missing).",
            len(uea_missing),
        )
        bulk_failed |= not _download_bulk(cache_dir, UEA_BULK_URL)
        uea_missing = {n for n in uea_missing if not _is_cached(cache_dir, n)}

    for name in missing:
        folder = cache_dir / name
        # sktime skips its download whenever the folder exists, so this dataset
        # would fail on every run until someone removes the folder.
        if folder.is_dir() and not _is_cached(cache_dir, name):
            logger.warning(
                "Dataset folder %s exists without complete TRAIN/TEST .ts files; "
                "sktime will not download into it. Delete it to re-download.",
                folder,
            )

    remaining = sorted(ucr_missing | uea_missing)
    if remaining and bulk_failed:
        # The fast path is gone; say so, and say what it now costs, because the
        # run still succeeds and the only other sign is how long it takes.
        listing = ", ".join(remaining[:10]) + (" …" if len(remaining) > 10 else "")
        logger.warning(
            "Bulk archive download failed - sktime will download each of the %d "
            "missing dataset(s) when it is loaded, which is considerably slower: %s",
            len(remaining),
            listing,
        )
    elif remaining:
        logger.info("sktime downloads each missing dataset when it is loaded.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cached(cache_dir: Path, name: str) -> bool:
    """Return True if both TRAIN and TEST .ts files exist and are non-empty."""
    train = cache_dir / name / f"{name}_TRAIN.ts"
    test = cache_dir / name / f"{name}_TEST.ts"
    return (
        train.exists()
        and test.exists()
        and train.stat().st_size > 0
        and test.stat().st_size > 0
    )


def _zip_top_level_dir(zip_path: Path) -> str | None:
    """
    Return the single top-level directory name inside *zip_path*, or ``None``.

    Reads only the central directory (``namelist()``); does not extract anything.
    Returns ``None`` when the zip has multiple top-level directories or no
    directory structure at all.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        tops = {n.split("/")[0] for n in zf.namelist() if "/" in n}
    return tops.pop() if len(tops) == 1 else None


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    logger.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tqdm.tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=dest.name,
        leave=True,
    ) as bar:

        def _hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0 and bar.total is None:
                bar.total = total_size
            bar.update(block_size)

        urllib.request.urlretrieve(url, dest, reporthook=_hook)

    logger.info("Saved to %s", dest)


def _extract_and_flatten(zip_path: Path, cache_dir: Path) -> None:
    """
    Extract *zip_path* into *cache_dir* and remove the single wrapper directory.

    Bulk archives wrap all dataset subdirectories under one top-level folder
    (e.g. ``Univariate_ts/`` or ``Multivariate_ts/``).  The inner directory
    name is detected automatically from the zip's central directory so that
    naming variations across archive versions are handled transparently.

    After flattening the layout matches what sktime expects::

        cache_dir/{name}/{name}_TRAIN.ts

    Datasets that are already present at the top level are skipped.
    """
    inner_prefix = _zip_top_level_dir(zip_path)
    if inner_prefix is None:
        logger.warning(
            "Could not detect a single top-level directory in %s — extracting as-is.",
            zip_path.name,
        )

    logger.info("Extracting %s …", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    if inner_prefix is None:
        return

    wrapper = cache_dir / inner_prefix
    if not wrapper.is_dir():
        logger.warning(
            "Wrapper directory '%s' not found after extraction.", inner_prefix
        )
        return

    moved = 0
    for dataset_dir in list(wrapper.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dest = cache_dir / dataset_dir.name
        if dest.exists():
            shutil.rmtree(dataset_dir)
        else:
            dataset_dir.rename(dest)
            moved += 1

    shutil.rmtree(wrapper, ignore_errors=True)
    logger.info("Extracted %d dataset(s) from '%s'.", moved, inner_prefix)


def _download_bulk(cache_dir: Path, url: str) -> bool:
    """
    Download the bulk archive zip, extract+flatten, then delete the zip.

    Parameters
    ----------
    cache_dir:
        Directory the extracted datasets are written to.
    url:
        Archive URL to fetch.

    Returns
    -------
    bool
        ``True`` if the archive was fetched and extracted, ``False`` if it failed.
        A failure is recoverable: sktime downloads each missing dataset on load.
    """
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = cache_dir / f"_tmp_{zip_name}"
    try:
        _download_file(url, zip_path)
        _extract_and_flatten(zip_path, cache_dir)
    except Exception as exc:
        # Recoverable: sktime fetches the datasets one by one on load, so this is a
        # warning rather than an error, with the traceback kept for --debug.
        logger.warning("Bulk archive download failed for %s: %s", url, exc)
        logger.debug("Bulk download traceback", exc_info=True)
        return False
    else:
        return True
    finally:
        if zip_path.exists():
            zip_path.unlink()
