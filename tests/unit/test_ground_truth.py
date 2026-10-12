"""Unit tests for the ground-truth region schema and its masks."""

import numpy as np
import pytest
import torch

from xai4tsc.evaluation._ground_truth import (
    GroundTruthRegion,
    coefficient_mask,
    regions_for_class,
    regions_for_sample,
    time_mask,
)
from xai4tsc.utils import STFTransform


@pytest.fixture(scope="module")
def published_test_split():
    """
    Generate the shipped dataset's test split, with its ground truth scoped.

    These assertions used to read a committed copy of the dataset. The dataset is
    generated now, so they generate it: the numbers below are properties of the
    data itself (600 samples, 75 of them with no region, ~38 % coverage), not of a
    file that happened to be in the checkout.
    """
    from xai4tsc.data.datasets import SynthMultilabelFreqDataset

    ds = SynthMultilabelFreqDataset()
    splits, encoder = ds.split(train_split=0.7, val_split=0.15, encode="multihot")
    return ds, splits[1], encoder


# ── the time-first case ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_interval_alone_is_enough():
    """A time-domain dataset supplies pos and len and nothing else."""
    region = GroundTruthRegion.from_mapping({"pos": 2, "len": 3})

    assert region.pos == 2
    assert region.length == 3
    assert region.channels is None, "an omitted scope means every channel"
    assert region.classes is None
    assert region.freqs == ()


@pytest.mark.unit
def test_time_mask_fills_the_half_open_interval():
    region = GroundTruthRegion.from_mapping({"pos": 2, "len": 3})

    mask = time_mask([region], n_channels=1, signal_length=8)

    assert mask.dtype == bool
    assert mask[0].tolist() == [False, False, True, True, True, False, False, False]


@pytest.mark.unit
def test_time_mask_broadcasts_over_channels_but_honours_a_scope():
    everywhere = GroundTruthRegion.from_mapping({"pos": 0, "len": 2})
    scoped = GroundTruthRegion.from_mapping({"pos": 0, "len": 2, "channels": [1]})

    assert time_mask([everywhere], 2, 4)[:, :2].all()
    only_second = time_mask([scoped], 2, 4)
    assert not only_second[0].any()
    assert only_second[1, :2].all()


@pytest.mark.unit
def test_time_mask_clips_and_ignores_out_of_range():
    over_end = GroundTruthRegion.from_mapping({"pos": 6, "len": 99})
    out_of_range_channel = GroundTruthRegion.from_mapping(
        {"pos": 0, "len": 2, "channels": [7]}
    )

    assert time_mask([over_end], 1, 8)[0].tolist()[6:] == [True, True]
    assert not time_mask([out_of_range_channel], 1, 8).any()


@pytest.mark.unit
def test_an_omitted_interval_covers_the_whole_series():
    """A region localized only in frequency has no time extent of its own."""
    region = GroundTruthRegion.from_mapping({"freq": 7})

    assert region.length is None
    assert region.interval(32) == (0, 32)
    assert time_mask([region], 1, 8).all()


# ── normalisation of stored shapes ────────────────────────────────────────────


@pytest.mark.unit
def test_singular_keys_normalise_to_tuples():
    """The shipped dataset writes `channel`, `freq` and `phase` in the singular."""
    region = GroundTruthRegion.from_mapping(
        {"pos": 0, "len": 1, "channel": 0, "freq": 19, "phase": 4.5}
    )

    assert region.channels == (0,)
    assert region.freqs == (19.0,)
    assert region.phases == (4.5,)


@pytest.mark.unit
def test_plural_keys_are_taken_as_given():
    region = GroundTruthRegion.from_mapping(
        {"pos": 0, "len": 1, "channels": [0, 2], "classes": [1], "freqs": [7, 19]}
    )

    assert region.channels == (0, 2)
    assert region.classes == (1,)
    assert region.freqs == (7.0, 19.0)


@pytest.mark.unit
def test_flat_list_and_per_class_dict_are_both_accepted():
    flat = regions_for_sample({"ground_truth": [{"freq": 7}, {"freq": 19}]})
    per_class = regions_for_sample(
        {"ground_truth": {0: [{"freq": 7}], 1: [{"freq": 19}]}}
    )

    assert [r.freqs for r in flat] == [(7.0,), (19.0,)]
    assert [r.freqs for r in per_class] == [(7.0,), (19.0,)]
    # The outer key names the class the region discriminates.
    assert [r.classes for r in per_class] == [(0,), (1,)]
    assert [r.classes for r in flat] == [None, None]


@pytest.mark.unit
@pytest.mark.parametrize("spelling", ["len", "length"])
def test_both_length_spellings_are_accepted(spelling):
    """Datasets store `len`; the dataclass field is `length`. Neither should trap."""
    region = GroundTruthRegion.from_mapping({"pos": 0, spelling: 5})

    assert region.interval(100) == (0, 5)


@pytest.mark.unit
@pytest.mark.parametrize("bad_key", ["lenght", "poss", "colour", "ground_truth"])
def test_an_unrecognised_field_raises(bad_key):
    """
    An ignored key would be worse than an error here.

    Every field is optional and an omitted interval means "the whole series", so
    silently dropping a misspelled `len` would widen the mask to the entire sample
    instead of failing - a wrong score rather than a stack trace.
    """
    with pytest.raises(ValueError, match="unrecognised field"):
        GroundTruthRegion.from_mapping({"pos": 0, "len": 5, bad_key: 1})


# ── absent versus malformed ───────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "meta",
    [None, {}, {"other": 1}, {"ground_truth": None}, {"ground_truth": []}],
    ids=["none", "empty", "no-key", "null-gt", "empty-gt"],
)
def test_absent_ground_truth_yields_no_regions(meta):
    """A sample may legitimately have no discriminative region."""
    assert regions_for_sample(meta) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "meta",
    [
        {"ground_truth": 5},
        {"ground_truth": [42]},
        {"ground_truth": [{"channels": [0]}]},
        {"ground_truth": [{"pos": "nope", "len": 1}]},
        {"ground_truth": {0: 5}},
    ],
    ids=[
        "scalar-gt",
        "non-mapping-region",
        "localizes-nothing",
        "bad-pos",
        "bad-class-value",
    ],
)
def test_malformed_ground_truth_raises(meta):
    """
    Malformed metadata must not read as "no ground truth".

    Six localisation metrics score against these regions, so silently yielding an
    empty list would turn a typo into a number instead of an error.
    """
    with pytest.raises(ValueError):
        regions_for_sample(meta)


# ── the coefficient grid, unchanged ───────────────────────────────────────────


@pytest.mark.unit
def test_coefficient_mask_frequency_grid():
    regions = regions_for_sample({"ground_truth": {0: [{"channel": 0, "freq": 7}]}})

    mask = coefficient_mask(regions, (1, 17), signal_length=32, transform=None)

    assert mask.sum() == 1
    assert mask[0, 7]


@pytest.mark.unit
def test_coefficient_mask_time_frequency_grid():
    stft = STFTransform(n_fft=16, win_length=16, hop_length=8)
    stft.forward(torch.randn(1, 1, 64))
    regions = regions_for_sample(
        {"ground_truth": [{"channel": 0, "pos": 16, "len": 16, "freq": 4}]}
    )

    mask = coefficient_mask(regions, (1, 9, 9), signal_length=64, transform=stft)

    # freq bin round(4*16/64)=1; frames overlapping [16,32) with hop 8 -> 2..4.
    assert mask.any() and not mask.all()
    assert mask[0, 1, 2] and mask[0, 1, 3]


@pytest.mark.unit
def test_multiple_freqs_mark_multiple_bins():
    regions = regions_for_sample({"ground_truth": [{"freqs": [3, 7]}]})

    mask = coefficient_mask(regions, (1, 17), signal_length=32, transform=None)

    assert mask.sum() == 2
    assert mask[0, 3] and mask[0, 7]


# ── frequency units: cycles per series, or Hz plus a sampling rate ────────────


@pytest.mark.unit
def test_a_frequency_without_a_sampling_rate_is_cycles_per_series():
    """The unit a coefficient grid is indexed by, and the back-compatible default."""
    region = GroundTruthRegion(freqs=(19.0,))

    assert region.cycles(500) == (19.0,)
    mask = coefficient_mask([region], (1, 251), signal_length=500, transform=None)
    assert mask.sum() == 1
    assert mask[0, 19]


@pytest.mark.unit
def test_a_frequency_in_hz_is_converted_against_the_series_duration():
    """
    With a sampling rate the frequency is Hz, so it scales by the duration.

    500 samples at 100 Hz is a 5 s series, so 19 Hz completes 95 oscillations
    across it and belongs in rFFT bin 95. Reading the stored 19 as a bin index
    instead is the defect this guards: it masks 3.8 Hz, where no class signal is,
    and the metric then reports chance for every explainer.
    """
    region = GroundTruthRegion(freqs=(19.0,), sampling_rate=100.0)

    assert region.cycles(500) == (95.0,)
    mask = coefficient_mask([region], (1, 251), signal_length=500, transform=None)
    assert mask.sum() == 1
    assert mask[0, 95]


@pytest.mark.unit
def test_a_sampling_rate_on_the_sample_row_reaches_every_region():
    """Stating the rate once per sample must scope it over that sample's regions."""
    regions = regions_for_sample(
        {
            "ground_truth": [
                {"pos": 0, "len": 150, "freq": 7},
                {"pos": 200, "len": 150, "freq": 37},
            ],
            "sampling_rate": 100,
        }
    )

    assert [r.sampling_rate for r in regions] == [100.0, 100.0]
    assert [r.cycles(500) for r in regions] == [(35.0,), (185.0,)]


@pytest.mark.unit
def test_a_regions_own_sampling_rate_wins_over_the_rows():
    regions = regions_for_sample(
        {
            "ground_truth": [{"freq": 7, "sampling_rate": 50}, {"freq": 7}],
            "sampling_rate": 100,
        }
    )

    assert [r.cycles(500) for r in regions] == [(70.0,), (35.0,)]


@pytest.mark.unit
def test_hz_scales_onto_the_time_frequency_grid_too():
    stft = STFTransform(n_fft=16, win_length=16, hop_length=8)
    stft.forward(torch.randn(1, 1, 64))
    regions = regions_for_sample(
        {
            "ground_truth": [{"channel": 0, "pos": 16, "len": 16, "freq": 8}],
            "sampling_rate": 32,
        }
    )

    mask = coefficient_mask(regions, (1, 9, 9), signal_length=64, transform=stft)

    # 8 Hz over 64 samples at 32 Hz is 16 cycles/series -> STFT bin round(16*16/64)=4.
    assert mask[0, 4, 2] and mask[0, 4, 3]
    assert not mask[0, 1].any(), "the pre-conversion bin must no longer be marked"


# ── the shipped dataset still parses and masks identically ────────────────────


@pytest.mark.unit
def test_every_generated_region_parses(published_test_split):
    """
    The generator's own output must parse as it stands.

    This is the guard on `from_mapping`: the regions it writes use the singular
    field names, and a normalisation slip would change every mask silently.
    """
    _ds, (_x, _y, metadata), _encoder = published_test_split
    parsed = [regions_for_sample(row) for row in metadata]

    assert len(parsed) == 600
    assert sum(1 for r in parsed if not r) == 75, (
        "75 samples legitimately carry no region; they must parse as empty, not raise"
    )
    flat = [region for regions in parsed for region in regions]
    assert flat, "the rest must yield regions"
    assert all(r.channels == (0,) for r in flat), "this dataset is univariate"
    assert all(len(r.freqs) == 1 for r in flat), "one frequency component per region"
    assert all(r.length == 150 for r in flat), "fixed-width bursts"
    assert all(r.sampling_rate == 100 for r in flat), (
        "the row's sampling rate must reach every region, or its frequency in Hz "
        "is read as cycles per series and the mask lands on the wrong bins"
    )


@pytest.mark.unit
def test_regions_mask_the_expected_time_fraction(published_test_split):
    """
    Coverage sets the chance level, so a score has to be read against it.

    A mean of ~38 % means a random attribution scores ~0.38 on Pointing Game, not
    0.5; pinning it here keeps that number discoverable.
    """
    _ds, (_x, _y, metadata), _encoder = published_test_split
    coverage = np.array(
        [time_mask(regions_for_sample(row), 1, 500).mean() for row in metadata]
    )

    assert coverage.max() < 1.0, "a fully covered sample would make the metric vacuous"
    assert 0.35 < coverage.mean() < 0.42


# ── s_batch: what the localisation metrics are handed ─────────────────────────


@pytest.fixture
def localisation_setup():
    """Build a model and a real explanation for driving a localisation metric."""
    from sklearn.preprocessing import LabelEncoder
    from torch import nn

    from xai4tsc.xai.explain import _get_explanation

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(24, 2)

        def forward(self, x):
            return self.fc(x.flatten(1))

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1, 24)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    model = _Tiny().eval()
    exp = _get_explanation(
        method="Integrated_Gradients",
        model=model,
        data=data,
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        params={"n_steps": 8},
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )
    return model, exp


_CHEAP = {
    "disable_warnings": True,
    "display_progressbar": False,
    "return_aggregate": True,
}


@pytest.mark.unit
def test_s_batch_is_single_channel_and_matches_the_input(localisation_setup):
    """Quantus asserts s_batch.shape[1] == 1, so channels are unioned."""
    from xai4tsc.evaluation.base import QuantusEvaluator

    _, exp = localisation_setup
    exp.metadata = [{"ground_truth": [{"pos": 2, "len": 4, "channels": [0]}]}] * 4

    s_batch, keep = QuantusEvaluator("Pointing Game")._ground_truth_batch(exp, exp.data)

    assert s_batch.shape == (4, 1, 24)
    assert s_batch.dtype == bool
    assert keep.all()
    assert s_batch[0, 0, 2:6].all()
    assert not s_batch[0, 0, 6:].any()


@pytest.mark.unit
def test_samples_without_a_region_are_dropped_not_scored(localisation_setup):
    """
    An all-False mask would score 0 and look like a result.

    Quantus cannot catch it — its guard, `len(np.nonzero(s_batch)) > 0`, counts
    dimensions rather than elements, so an empty mask passes straight through.
    """
    from xai4tsc.evaluation.base import QuantusEvaluator

    _, exp = localisation_setup
    exp.metadata = [
        {"ground_truth": [{"pos": 0, "len": 6}]},
        {"ground_truth": []},
        {"ground_truth": [{"pos": 8, "len": 4}]},
        {"ground_truth": []},
    ]

    s_batch, keep = QuantusEvaluator("Pointing Game")._ground_truth_batch(exp, exp.data)

    assert keep.tolist() == [True, False, True, False]
    assert s_batch.shape[0] == 2, "only the samples with a region are handed over"
    assert s_batch.any(axis=(1, 2)).all(), "no empty mask reaches the metric"


@pytest.mark.unit
def test_quantus_own_empty_mask_guard_is_vacuous():
    """Documents why the exclusion above has to exist on our side."""
    empty = np.zeros((4, 1, 24), dtype=bool)

    assert len(np.nonzero(empty)) > 0, "the library's guard passes an all-zero mask"
    assert not empty.any(), "...even though it contains nothing"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "metadata"),
    [
        ("no metadata at all", None),
        ("metadata but no regions", [{"ground_truth": []}] * 4),
    ],
)
def test_nothing_to_localise_against_raises(localisation_setup, label, metadata):
    """Better a stated error than a score computed against an empty mask."""
    from xai4tsc.evaluation.base import QuantusEvaluator

    _, exp = localisation_setup
    exp.metadata = metadata

    with pytest.raises(ValueError, match=r"ground-truth|no metadata"):
        QuantusEvaluator("Pointing Game")._ground_truth_batch(exp, exp.data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "metric",
    [
        "AUC",
        "Attribution Localisation",
        "Pointing Game",
        "Relevance Mass Accuracy",
        "Relevance Rank Accuracy",
        "Top-K Intersection",
    ],
)
def test_every_localisation_metric_scores_with_ground_truth(localisation_setup, metric):
    """All six were raising `s_batch should be of type np.ndarray` before this."""
    from xai4tsc.evaluation.evaluate import evaluate

    model, exp = localisation_setup
    exp.metadata = [{"ground_truth": [{"pos": 4, "len": 8}]}] * 4

    score = evaluate(
        model,
        metric,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params=_CHEAP,
        metric_call_params={"softmax": True},
    )

    assert score is not None
    assert np.all(np.isfinite(np.asarray(score, dtype=float)))


@pytest.mark.unit
def test_a_non_localisation_metric_gets_no_s_batch(localisation_setup):
    """Only the localisation group needs masks; nothing else should see them."""
    from xai4tsc.evaluation.evaluate import evaluate

    model, exp = localisation_setup
    exp.metadata = None  # would raise if s_batch were built

    score = evaluate(
        model,
        "Complexity",
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params=_CHEAP,
    )

    assert score is not None


# ── skipping where no ground truth exists ─────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "metric", ["Complexity", "Pixel-Flipping", "Max-Sensitivity", "Time-Frequency AUC"]
)
def test_metrics_that_need_no_ground_truth_are_never_blocked(
    localisation_setup, metric
):
    """Only the localisation group has this requirement."""
    from xai4tsc.evaluation import ground_truth_missing

    _, exp = localisation_setup
    exp.metadata = None

    assert ground_truth_missing(metric, exp) is None


@pytest.mark.unit
def test_a_dataset_without_ground_truth_is_an_unmet_requirement(localisation_setup):
    """Most datasets declare none; that is a skip, not a failure."""
    from xai4tsc.evaluation import ground_truth_missing

    _, exp = localisation_setup
    exp.metadata = None

    reason = ground_truth_missing("Pointing Game", exp)

    assert reason is not None
    assert "declares none" in reason


@pytest.mark.unit
def test_regions_present_means_nothing_is_missing(localisation_setup):
    from xai4tsc.evaluation import ground_truth_missing

    _, exp = localisation_setup
    exp.metadata = [{"ground_truth": [{"pos": 0, "len": 4}]}] * 4

    assert ground_truth_missing("Pointing Game", exp) is None


@pytest.mark.unit
def test_malformed_metadata_is_not_reported_as_a_skip(localisation_setup):
    """
    A typo must not masquerade as "this dataset has no ground truth".

    The skip path and the defect path lead to very different places — one is an
    expected row in the results, the other needs fixing — so the predicate lets
    the error through rather than swallowing it.
    """
    from xai4tsc.evaluation import ground_truth_missing

    _, exp = localisation_setup
    exp.metadata = [{"ground_truth": [{"pos": 0, "lenght": 4}]}]

    with pytest.raises(ValueError, match="unrecognised field"):
        ground_truth_missing("Pointing Game", exp)


@pytest.mark.unit
def test_evaluate_skips_rather_than_raising(localisation_setup, caplog):
    """End to end: no ground truth yields None and a logged reason, not an error."""
    from xai4tsc.evaluation.evaluate import evaluate

    model, exp = localisation_setup
    exp.metadata = None

    with caplog.at_level("WARNING"):
        score = evaluate(
            model,
            "Pointing Game",
            explanation=exp,
            data=exp.data,
            labels=exp.labels,
            metric_class_params=_CHEAP,
        )

    assert score is None
    assert "Pointing Game" in caplog.text
    assert "ground-truth" in caplog.text


# ── per-class scoping ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_regions_for_class_keeps_only_that_class():
    regions = regions_for_sample(
        {
            "ground_truth": [
                {"pos": 0, "len": 4, "freq": 7, "classes": [4, 5]},
                {"pos": 8, "len": 4, "freq": 19, "classes": [2, 3]},
            ]
        }
    )

    assert [r.freqs for r in regions_for_class(regions, 4)] == [(7.0,)]
    assert [r.freqs for r in regions_for_class(regions, 2)] == [(19.0,)]
    assert regions_for_class(regions, 7) == [], "a class with no evidence here"


@pytest.mark.unit
def test_an_unscoped_region_applies_to_every_class():
    """A single-label dataset need not say which class; the sample's label does."""
    regions = regions_for_sample({"ground_truth": [{"pos": 0, "len": 4}]})

    assert len(regions_for_class(regions, 0)) == 1
    assert len(regions_for_class(regions, 99)) == 1


@pytest.mark.unit
def test_no_recorded_target_keeps_everything():
    """Without a recorded target the union is the best available answer."""
    regions = regions_for_sample(
        {"ground_truth": [{"pos": 0, "len": 4, "classes": [1]}]}
    )

    assert len(regions_for_class(regions, None)) == 1


@pytest.mark.unit
def test_each_frequency_is_scoped_to_the_classes_it_marks(published_test_split):
    """
    A label bit of this dataset means "a burst at that frequency is present".

    The regions carry only the frequency, so the dataset derives which encoded
    classes each one marks - every combination whose bit is set.
    """
    ds, (_x, _y, metadata), encoder = published_test_split

    expected = {
        freq: tuple(k for k, combo in enumerate(encoder.classes_) if combo[bit])
        for bit, freq in enumerate(ds.discriminative_frequencies)
    }
    scoped = [r for row in metadata for r in row["ground_truth"]]

    assert scoped
    assert all(tuple(r["classes"]) == expected[r["freq"]] for r in scoped)


@pytest.mark.unit
def test_each_class_gets_its_own_mask_on_a_multi_label_sample(published_test_split):
    """
    The whole point of the per-class scope.

    The dataset overlaps co-occurring class windows by 25-50 % by design, so
    scoring every class against the union would discard exactly what it was built
    to measure - and give each class an identical mask.
    """
    _ds, (_x, _y, metadata), encoder = published_test_split

    # A sample carrying two discriminative bursts.
    row = next(r for r in metadata if len(r["ground_truth"]) == 2)
    regions = regions_for_sample(row)
    masks = {
        k: time_mask(regions_for_class(regions, k), 1, 500).tobytes()
        for k in range(len(encoder.classes_))
    }

    assert len(set(masks.values())) > 1, "a per-class mask must depend on the class"
    # The two single-frequency classes see one packet each, the pair sees both.
    coverage = {
        k: time_mask(regions_for_class(regions, k), 1, 500).mean() for k in masks
    }
    assert min(coverage.values()) == 0.0, "a class with no packet here scores nothing"
    assert max(coverage.values()) > 0.4, "the class with both packets covers the most"


@pytest.mark.unit
def test_explained_targets_are_recorded(localisation_setup):
    """A ground-truth metric cannot pick the right region without them."""
    _, exp = localisation_setup

    assert exp.explained_targets is not None
    assert exp.explained_targets.tolist() == exp.labels.tolist(), (
        "targets='label' explains each sample's own class"
    )


# ── collapsing the channel axis ───────────────────────────────────────────────


@pytest.fixture
def multichannel_setup():
    """Build a 3-channel explanation for exercising the channel collapse."""
    from sklearn.preprocessing import LabelEncoder
    from torch import nn

    from xai4tsc.xai.explain import _get_explanation

    class _Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 16, 2)

        def forward(self, x):
            return self.fc(x.flatten(1))

    torch.manual_seed(0)
    data = np.random.default_rng(0).standard_normal((2, 3, 16)).astype(np.float32)
    labels = np.array([0, 1])
    model = _Wide().eval()
    exp = _get_explanation(
        method="Integrated_Gradients",
        model=model,
        data=data,
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        params={"n_steps": 4},
        targets="label",
        orig_indices=np.arange(2),
        device="cpu",
    )
    return model, exp


def _batch(exp):
    """Run the ground-truth batch builder for a localisation metric."""
    from xai4tsc.evaluation.base import QuantusEvaluator

    return QuantusEvaluator("Pointing Game")._ground_truth_batch(exp, exp.data)


@pytest.mark.unit
def test_a_channel_scoped_region_survives_the_collapse(multichannel_setup):
    """
    Building the mask at one channel would drop it rather than fold it in.

    `applies_to_channel` is asked about channel 0 only, so a region scoped to
    channel 1 would match nothing and vanish — a silently empty mask, which is the
    failure this whole path is built to avoid.
    """
    _, exp = multichannel_setup
    exp.metadata = [{"ground_truth": [{"pos": 2, "len": 4, "channels": [1]}]}] * 2

    s_batch, _ = _batch(exp)

    assert s_batch.shape == (2, 1, 16)
    assert s_batch[0, 0, 2:6].all(), "the region must survive, folded onto one channel"


@pytest.mark.unit
def test_collapsing_a_channel_scoped_region_warns(multichannel_setup, caplog):
    """The mask can no longer say which channel carried the signal."""
    _, exp = multichannel_setup
    exp.metadata = [{"ground_truth": [{"pos": 2, "len": 4, "channels": [1]}]}] * 2

    with caplog.at_level("WARNING"):
        _batch(exp)

    assert "unioned across the 3 channels" in caplog.text
    assert "Pointing Game" in caplog.text


@pytest.mark.unit
def test_no_warning_when_the_collapse_loses_nothing(multichannel_setup, caplog):
    """A region that applies to every channel collapses without losing anything."""
    _, exp = multichannel_setup
    exp.metadata = [{"ground_truth": [{"pos": 2, "len": 4}]}] * 2

    with caplog.at_level("WARNING"):
        _batch(exp)

    assert "unioned across" not in caplog.text


@pytest.mark.unit
def test_no_warning_for_univariate_data(localisation_setup, caplog):
    """Nothing to collapse when there is one channel; the common case stays quiet."""
    _, exp = localisation_setup
    exp.metadata = [{"ground_truth": [{"pos": 2, "len": 4}]}] * 4

    with caplog.at_level("WARNING"):
        _batch(exp)

    assert "unioned across" not in caplog.text
