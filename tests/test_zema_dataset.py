import os
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as hst

from zema_emc_annotated import dataset
from zema_emc_annotated.data_types import UncertainArray
from zema_emc_annotated.dataset import (
    ExtractionDataType,
    LOCAL_ZEMA_DATASET_PATH,
    ZEMA_DATASET_HASH,
    ZEMA_DATASET_URL,
    ZEMA_QUANTITIES,
    ZeMASamples,
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


def test_dataset_has_attribute_LOCAL_ZEMA_DATASET_PATH() -> None:
    assert hasattr(dataset, "LOCAL_ZEMA_DATASET_PATH")


def test_dataset_attribute_LOCAL_ZEMA_DATASET_PATH_is_path() -> None:
    assert isinstance(LOCAL_ZEMA_DATASET_PATH, Path)


def test_dataset_attribute_LOCAL_ZEMA_DATASET_PATH_in_all() -> None:
    assert "LOCAL_ZEMA_DATASET_PATH" in dataset.__all__


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


def test_zema_samples_is_callable() -> None:
    assert callable(ZeMASamples)


def test_dataset_all_contains_zema_samples() -> None:
    assert ZeMASamples.__name__ in dataset.__all__


def test_zema_samples_has_docstring() -> None:
    assert ZeMASamples.__doc__ is not None


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
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_store_cache_stores_pickle_file_for_random_input(size_scaler: int) -> None:
    zema_samples = ZeMASamples(11, size_scaler)
    assert os.path.exists(
        zema_samples._cache_path(signature(ZeMASamples).parameters["normalize"].default)
    )


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10), hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_check_and_load_cache_runs_for_random_uncertain_values_and_returns(
    n_samples: int, size_scaler: int
) -> None:
    result = ZeMASamples(n_samples, size_scaler)._check_and_load_cache(
        signature(ZeMASamples).parameters["normalize"].default
    )
    assert result is None or isinstance(result, UncertainArray)


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_check_and_load_cache_returns_something_for_existing_file(
    size_scaler: int,
) -> None:
    zema_samples = ZeMASamples(12, size_scaler)
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


def test_dataset_extract_samples_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(ZeMASamples).parameters


def test_dataset_extract_samples_expects_parameter_size_scaler() -> None:
    assert "size_scaler" in signature(ZeMASamples).parameters


def test_dataset_extract_samples_expects_parameter_n_samples_as_int() -> None:
    assert signature(ZeMASamples).parameters["n_samples"].annotation is int


def test_dataset_zema_samples_expects_parameter_size_scaler_as_int() -> None:
    assert signature(ZeMASamples).parameters["size_scaler"].annotation is int


def test_dataset_extract_samples_parameter_n_samples_default_is_one() -> None:
    assert signature(ZeMASamples).parameters["n_samples"].default == 1


def test_dataset_extract_samples_parameter_size_scaler_default_is_one() -> None:
    assert signature(ZeMASamples).parameters["size_scaler"].default == 1


def test_dataset_zema_samples_states_uncertain_values_are_uncertain_array() -> None:
    assert ZeMASamples.__annotations__["uncertain_values"] is UncertainArray


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array(n_samples: int) -> None:
    assert isinstance(ZeMASamples(n_samples).uncertain_values, UncertainArray)


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array_with_n_samples_values(
    n_samples: int,
) -> None:
    assert len(ZeMASamples(n_samples).values) == n_samples


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array_with_n_samples_uncertainties(
    n_samples: int,
) -> None:
    result_uncertainties = ZeMASamples(n_samples).uncertainties
    assert result_uncertainties is not None
    assert len(result_uncertainties) == n_samples


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_default_extract_samples_returns_values_of_eleven_sensors(
    n_samples: int,
) -> None:
    assert ZeMASamples(n_samples).values.shape[1] == 11


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10), hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_eleven_times_scaler_values(
    n_samples: int, size_scaler: int
) -> None:
    assert ZeMASamples(n_samples, size_scaler).values.shape[1] == 11 * size_scaler


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_default_extract_samples_returns_uncertainties_of_eleven_sensors(
    n_samples: int,
) -> None:
    result_uncertainties = ZeMASamples(n_samples).uncertainties
    assert result_uncertainties is not None
    assert result_uncertainties.shape[1] == 11


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10), hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_eleven_times_scaler_uncertainties(
    n_samples: int, size_scaler: int
) -> None:
    result_uncertainties = ZeMASamples(n_samples, size_scaler).uncertainties
    assert result_uncertainties is not None
    assert result_uncertainties.shape[1] == 11 * size_scaler


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_values_and_uncertainties_which_are_not_similar(
    n_samples: int,
) -> None:
    result = ZeMASamples(n_samples)
    assert not np.all(result.values == result.uncertainties)


@pytest.mark.webtest
def test_zema_samples_fails_for_more_than_4766_samples() -> None:
    with pytest.raises(
        ValueError,
        match=r"all the input array dimensions except for the concatenation axis must "
        r"match exactly.*",
    ):
        ZeMASamples(4767)


@pytest.mark.webtest
def test_zema_samples_creates_pickle_files() -> None:
    for size_scaler in (1, 10, 100, 1000, 2000):
        for normalize in (True, False):
            assert ZeMASamples(size_scaler=size_scaler, normalize=normalize)


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10), hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_zema_samples_normalized_mean_is_smaller_or_equal(
    n_samples: int, size_scaler: int
) -> None:
    normalized_result = ZeMASamples(n_samples, size_scaler, True)
    not_normalized_result = ZeMASamples(n_samples, size_scaler)
    assert not_normalized_result.values.mean() >= normalized_result.values.mean()


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10), hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_zema_samples_normalized_std_is_smaller_or_equal(
    n_samples: int, size_scaler: int
) -> None:
    normalized_result = ZeMASamples(n_samples, size_scaler, True)
    not_normalized_result = ZeMASamples(n_samples, size_scaler)
    assert not_normalized_result.values.std() >= normalized_result.values.std()
