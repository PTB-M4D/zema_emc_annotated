from inspect import isclass, signature
from os.path import exists
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy

from zema_emc_annotated import dataset
from zema_emc_annotated.data_types import SampleSize, UncertainArray
from zema_emc_annotated.dataset import (
    ExtractionDataType,
    ZEMA_DATASET_HASH,
    ZEMA_DATASET_URL,
    ZEMA_QUANTITIES,
    ZeMASamples,
)


@composite
def sample_sizes(draw: DrawFn) -> SearchStrategy[SampleSize]:
    small_positive_integers = hst.integers(min_value=1, max_value=10)
    return cast(
        SearchStrategy[SampleSize],
        SampleSize(
            idx_first_cycle=draw(small_positive_integers),
            n_cycles=draw(small_positive_integers),
            datapoints_per_cycle=draw(small_positive_integers),
        ),
    )


def test_dataset_has_docstring() -> None:
    assert dataset.__doc__ is not None


def test_dataset_has_enum_extraction_data() -> None:
    assert hasattr(dataset, "ExtractionDataType")


def test_extraction_data_enum_has_docstring_with_values() -> None:
    assert ExtractionDataType.__doc__ is not None
    assert "VALUES" in ExtractionDataType.__doc__


def test_extraction_data_enum_has_docstring_with_uncertainties() -> None:
    assert ExtractionDataType.__doc__ is not None
    assert "UNCERTAINTIES" in ExtractionDataType.__doc__


def test_dataset_extraction_data_contains_key_for_uncertainties() -> None:
    assert "qudt:standardUncertainty" in ExtractionDataType._value2member_map_


def test_dataset_extraction_data_contains_key_for_values() -> None:
    assert "qudt:value" in ExtractionDataType._value2member_map_


def test_dataset_extraction_data_contains_first_values_and_then_uncertainties() -> None:
    ordered_extraction_data_type = tuple(datatype for datatype in ExtractionDataType)
    assert "value" in ordered_extraction_data_type[0].value


def test_dataset_extraction_data_contains_uncertainties_at_second_position() -> None:
    ordered_extraction_data_type = tuple(datatype for datatype in ExtractionDataType)
    assert "Uncertainty" in ordered_extraction_data_type[1].value


def test_dataset_all_contains_extraction_data() -> None:
    assert ExtractionDataType.__name__ in dataset.__all__


def test_dataset_has_constant_quantities() -> None:
    assert hasattr(dataset, "ZEMA_QUANTITIES")


def test_dataset_constant_quantities_is_tuple() -> None:
    assert isinstance(ZEMA_QUANTITIES, tuple)


def test_dataset_constant_quantities_contains_acceleration() -> None:
    assert "Acceleration" in ZEMA_QUANTITIES


def test_dataset_constant_quantities_contains_active_current() -> None:
    assert "Active_Current" in ZEMA_QUANTITIES


def test_dataset_constant_quantities_contains_force() -> None:
    assert "Force" in ZEMA_QUANTITIES


def test_dataset_constant_quantities_contains_motor_current() -> None:
    assert "Motor_Current" in ZEMA_QUANTITIES


def test_dataset_constant_quantities_contains_pressure() -> None:
    assert "Pressure" in ZEMA_QUANTITIES


def test_dataset_constant_quantities_contains_sound_pressure() -> None:
    assert "Sound_Pressure" in ZEMA_QUANTITIES


def test_dataset_constant_quantities_contains_velocity() -> None:
    assert "Velocity" in ZEMA_QUANTITIES


def test_dataset_all_contains_constant_quantities() -> None:
    assert "ZEMA_QUANTITIES" in dataset.__all__


def test_dataset_has_attribute_ZEMA_DATASET_URL() -> None:
    assert hasattr(dataset, "ZEMA_DATASET_URL")


def test_dataset_attribute_ZEMA_DATASET_URL_is_string() -> None:
    assert isinstance(ZEMA_DATASET_URL, str)


def test_dataset_attribute_ZEMA_DATASET_URL_in_all() -> None:
    assert "ZEMA_DATASET_URL" in dataset.__all__


def test_dataset_has_attribute_ZEMA_DATASET_HASH() -> None:
    assert hasattr(dataset, "ZEMA_DATASET_HASH")


def test_dataset_attribute_ZEMA_DATASET_HASH() -> None:
    assert isinstance(ZEMA_DATASET_HASH, str)


def test_dataset_attribute_ZEMA_DATASET_HASH_in_all() -> None:
    assert "ZEMA_DATASET_HASH" in dataset.__all__


def test_dataset_has_attribute_zema_samples() -> None:
    assert hasattr(dataset, "ZeMASamples")


def test_zema_samples_is_class() -> None:
    assert isclass(ZeMASamples)


def test_zema_samples_expects_parameter_normalize() -> None:
    assert "normalize" in signature(ZeMASamples).parameters


def test_zema_samples_expects_parameter_normalize_of_type_bool() -> None:
    assert signature(ZeMASamples).parameters["normalize"].annotation is bool


def test_zema_samples_expects_parameter_normalize_default_is_false() -> None:
    assert signature(ZeMASamples).parameters["normalize"].default is False


def test_zema_samples_expects_parameter_sample_size() -> None:
    assert "sample_size" in signature(ZeMASamples).parameters


def test_zema_samples_expects_parameter_idx_start_of_type_int() -> None:
    assert signature(ZeMASamples).parameters["sample_size"].annotation is SampleSize


def test_zema_samples_parameter_sample_size_default_is_sample_size_default() -> None:
    assert signature(ZeMASamples).parameters["sample_size"].default == SampleSize()


def test_zema_samples_expects_parameter_skip_hash_check() -> None:
    assert "skip_hash_check" in signature(ZeMASamples).parameters


def test_zema_samples_expects_parameter_skip_hash_check_of_type_bool() -> None:
    assert signature(ZeMASamples).parameters["skip_hash_check"].annotation is bool


def test_zema_samples_expects_parameter_skip_hash_check_default_is_false() -> None:
    assert signature(ZeMASamples).parameters["skip_hash_check"].default is False


def test_dataset_all_contains_zema_samples() -> None:
    assert ZeMASamples.__name__ in dataset.__all__


def test_zema_samples_has_docstring() -> None:
    assert ZeMASamples.__doc__ is not None


def test_zema_samples_has_attribute_extract_data() -> None:
    assert hasattr(ZeMASamples, "_extract_data")


def test_dataset_extract_data_is_callable() -> None:
    assert callable(ZeMASamples._extract_data)


def test_extract_data_has_docstring() -> None:
    assert ZeMASamples._extract_data.__doc__ is not None


def test_extract_data_expects_parameter_normalize() -> None:
    assert "normalize" in signature(ZeMASamples._extract_data).parameters


def test_extract_data_expects_parameter_normalize_of_type_bool() -> None:
    assert (
        signature(ZeMASamples._extract_data).parameters["normalize"].annotation is bool
    )


def test_extract_data_expects_parameter_skip_hash_check() -> None:
    assert "skip_hash_check" in signature(ZeMASamples._extract_data).parameters


def test_zema_samples_extract_data_parameter_skip_hash_check_of_type_bool() -> None:
    assert (
        signature(ZeMASamples._extract_data).parameters["skip_hash_check"].annotation
        is bool
    )


def test_zema_samples_extract_data_parameter_skip_hash_check_default_is_false() -> None:
    assert (
        signature(ZeMASamples._extract_data).parameters["skip_hash_check"].default
        is True
    )


def test_zema_samples_has_attribute_check_and_load_cache() -> None:
    assert hasattr(ZeMASamples, "_check_and_load_cache")


def test_dataset_check_and_load_cache_is_callable() -> None:
    assert callable(ZeMASamples._check_and_load_cache)


def test_check_and_load_cache_has_docstring() -> None:
    assert ZeMASamples._check_and_load_cache.__doc__ is not None


def test_check_and_load_cache_expects_parameter_normalize() -> None:
    assert "normalize" in signature(ZeMASamples._check_and_load_cache).parameters


def test_zema_samples_has_attribute_cache_path() -> None:
    assert hasattr(ZeMASamples, "_cache_path")


def test_dataset_cache_path_is_callable() -> None:
    assert callable(ZeMASamples._cache_path)


def test_cache_path_expects_parameter_normalize() -> None:
    assert "normalize" in signature(ZeMASamples._cache_path).parameters


def test_check_and_load_cache_expects_parameter_normalize_as_bool() -> None:
    assert (
        signature(ZeMASamples._check_and_load_cache).parameters["normalize"].annotation
        is bool
    )


def test_cache_path_has_docstring() -> None:
    assert ZeMASamples._cache_path.__doc__ is not None


def test_cache_path_actually_returns_path() -> None:
    assert isinstance(
        ZeMASamples()._cache_path(
            signature(ZeMASamples).parameters["normalize"].default
        ),
        Path,
    )


def test_zema_samples_has_attribute_store_cache() -> None:
    assert hasattr(ZeMASamples, "_store_cache")


def test_dataset_store_cache_is_callable() -> None:
    assert callable(ZeMASamples._store_cache)


def test_store_cache_has_docstring() -> None:
    assert ZeMASamples._store_cache.__doc__ is not None


def test_store_cache_expects_parameter_normalize() -> None:
    assert "normalize" in signature(ZeMASamples._store_cache).parameters


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_store_cache_stores_pickle_file_for_random_input(
    sample_size: SampleSize,
) -> None:
    zema_samples = ZeMASamples(sample_size, skip_hash_check=True)
    assert exists(
        zema_samples._cache_path(signature(ZeMASamples).parameters["normalize"].default)
    )


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_check_and_load_cache_runs_for_random_uncertain_values_and_returns(
    sample_size: SampleSize,
) -> None:
    result = ZeMASamples(sample_size, skip_hash_check=True)._check_and_load_cache(
        signature(ZeMASamples).parameters["normalize"].default
    )
    assert result is None or isinstance(result, UncertainArray)


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_check_and_load_cache_returns_something_for_existing_file(
    sample_size: SampleSize,
) -> None:
    zema_samples = ZeMASamples(sample_size, skip_hash_check=True)
    assert (
        zema_samples._check_and_load_cache(
            signature(ZeMASamples).parameters["normalize"].default
        )
        is not None
    )


def test_store_cache_expects_parameter_normalize_as_bool() -> None:
    assert (
        signature(ZeMASamples._store_cache).parameters["normalize"].annotation is bool
    )


def test_cache_path_expects_parameter_normalize_as_bool() -> None:
    assert signature(ZeMASamples._cache_path).parameters["normalize"].annotation is bool


def test_cache_path_expects_stats_to_return_path() -> None:
    assert signature(ZeMASamples._cache_path).return_annotation is Path


def test_dataset_zema_samples_states_uncertain_values_are_uncertain_array() -> None:
    assert ZeMASamples.__annotations__["uncertain_values"] is UncertainArray


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array(
    sample_size: SampleSize,
) -> None:
    assert isinstance(
        ZeMASamples(sample_size, skip_hash_check=True).uncertain_values, UncertainArray
    )


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array_with_n_samples_values(
    sample_size: SampleSize,
) -> None:
    assert (
        len(ZeMASamples(sample_size, skip_hash_check=True).values)
        == sample_size.n_cycles
    )


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array_with_n_samples_uncertainties(
    sample_size: SampleSize,
) -> None:
    result_uncertainties = ZeMASamples(sample_size).uncertainties
    assert result_uncertainties is not None
    assert len(result_uncertainties) == sample_size.n_cycles


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_extract_samples_returns_eleven_times_scaler_values(
    sample_size: SampleSize,
) -> None:
    assert (
        ZeMASamples(sample_size, skip_hash_check=True).values.shape[1]
        == 11 * sample_size.datapoints_per_cycle
    )


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_extract_samples_returns_eleven_times_scaler_uncertainties(
    sample_size: SampleSize,
) -> None:
    result_uncertainties = ZeMASamples(sample_size, skip_hash_check=True).uncertainties
    assert result_uncertainties is not None
    assert result_uncertainties.shape[1] == 11 * sample_size.datapoints_per_cycle


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_extract_samples_returns_values_and_uncertainties_which_are_not_similar(
    sample_size: SampleSize,
) -> None:
    result = ZeMASamples(sample_size, skip_hash_check=True)
    assert not np.all(result.values == result.uncertainties)


@pytest.mark.webtest
@given(
    hst.integers(min_value=1, max_value=10000),
)
@settings(deadline=None)
def test_zema_samples_fails_for_more_than_4766_samples(
    n_samples_above_max: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"all the input array dimensions except for the concatenation axis must "
        r"match exactly.*",
    ):
        ZeMASamples(SampleSize(4766, n_samples_above_max), skip_hash_check=True)


@pytest.mark.webtest
def test_zema_samples_creates_pickle_files() -> None:
    for size_scaler in (1, 10, 100, 1000, 2000):
        for normalize in (True, False):
            assert ZeMASamples(
                SampleSize(datapoints_per_cycle=size_scaler),
                normalize=normalize,
                skip_hash_check=True,
            )


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_zema_samples_normalized_mean_is_smaller_or_equal(
    sample_size: SampleSize,
) -> None:
    normalized_result = ZeMASamples(sample_size, normalize=True, skip_hash_check=True)
    not_normalized_result = ZeMASamples(sample_size, skip_hash_check=True)
    assert not_normalized_result.values.mean() >= normalized_result.values.mean()


@pytest.mark.webtest
@given(sample_sizes())
@settings(deadline=None)
def test_zema_samples_normalized_std_is_smaller_or_equal(
    sample_size: SampleSize,
) -> None:
    normalized_result = ZeMASamples(sample_size, normalize=True, skip_hash_check=True)
    not_normalized_result = ZeMASamples(sample_size, skip_hash_check=True)
    assert not_normalized_result.values.std() >= normalized_result.values.std()


@pytest.mark.webtest
@given(
    sample_sizes(),
    hst.booleans(),
)
@settings(deadline=None)
def test_zema_samples_cache_path_contains_starting_from_for_larger_than_zero_startpoint(
    sample_size: SampleSize, normalize: bool
) -> None:
    zema_samples = ZeMASamples(sample_size, normalize, skip_hash_check=True)
    assert "_starting_from_" in str(zema_samples._cache_path(normalize))
