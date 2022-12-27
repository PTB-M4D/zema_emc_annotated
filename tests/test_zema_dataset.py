import os
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as hst

from zema_emc_annotated import dataset
from zema_emc_annotated.data_types import UncertainArray
from zema_emc_annotated.dataset import (
    _cache_path,
    _check_and_load_cache,
    _store_cache,
    ExtractionDataType,
    LOCAL_ZEMA_DATASET_PATH,
    provide_zema_samples,
    ZEMA_DATASET_HASH,
    ZEMA_DATASET_URL,
    ZEMA_DATATYPES,
    ZEMA_QUANTITIES,
)
from .conftest import uncertain_arrays


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


def test_dataset_all_contains_extraction_data() -> None:
    assert ExtractionDataType.__name__ in dataset.__all__


def test_dataset_has_constant_datatypes() -> None:
    assert hasattr(dataset, "ZEMA_DATATYPES")


def test_dataset_constant_datatypes_is_tuple() -> None:
    assert isinstance(ZEMA_DATATYPES, tuple)


def test_dataset_constant_datatypes_contains_uncertainties() -> None:
    assert "qudt:standardUncertainty" in ZEMA_DATATYPES


def test_dataset_constant_datatypes_contains_for_values() -> None:
    assert "qudt:value" in ZEMA_DATATYPES


def test_dataset_all_contains_constant_datatypes() -> None:
    assert "ZEMA_DATATYPES" in dataset.__all__


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


def test_dataset_has_attribute_extract_samples() -> None:
    assert hasattr(dataset, "provide_zema_samples")


def test_dataset_extract_samples_is_callable() -> None:
    assert callable(provide_zema_samples)


def test_dataset_all_contains_extract_samples() -> None:
    assert provide_zema_samples.__name__ in dataset.__all__


def test_extract_samples_has_docstring() -> None:
    assert provide_zema_samples.__doc__ is not None


def test_dataset_has_attribute_check_and_load_cache() -> None:
    assert hasattr(dataset, "_check_and_load_cache")


def test_dataset_check_and_load_cache_is_callable() -> None:
    assert callable(_check_and_load_cache)


def test_check_and_load_cache_has_docstring() -> None:
    assert _check_and_load_cache.__doc__ is not None


def test_check_and_load_cache_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(_check_and_load_cache).parameters


def test_check_and_load_cache_expects_parameter_n_samples_as_int() -> None:
    assert signature(_check_and_load_cache).parameters["n_samples"].annotation is int


def test_dataset_has_attribute_cache_path() -> None:
    assert hasattr(dataset, "_cache_path")


def test_dataset_cache_path_is_callable() -> None:
    assert callable(_cache_path)


def test_cache_path_has_docstring() -> None:
    assert _cache_path.__doc__ is not None


def test_cache_path_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(_cache_path).parameters


def test_cache_path_expects_parameter_n_samples_as_int() -> None:
    assert signature(_cache_path).parameters["n_samples"].annotation is int


@given(hst.integers())
def test_cache_path_actually_returns_path(integer: int) -> None:
    assert isinstance(_cache_path(integer), Path)


def test_dataset_has_attribute_store_cache() -> None:
    assert hasattr(dataset, "_store_cache")


def test_dataset_store_cache_is_callable() -> None:
    assert callable(_store_cache)


def test_store_cache_has_docstring() -> None:
    assert _store_cache.__doc__ is not None


def test_store_cache_expects_parameter_uncertain_values() -> None:
    assert "uncertain_values" in signature(_store_cache).parameters


@given(uncertain_arrays(length=11))
def test_store_cache_runs_for_random_uncertain_values(
    uncertain_array: UncertainArray,
) -> None:
    _store_cache(uncertain_array)
    assert os.path.exists(_cache_path(11))


@given(hst.integers())
def test_check_and_load_cache_runs_for_random_uncertain_values_and_returns(
    integer: int,
) -> None:
    result = _check_and_load_cache(integer)
    assert result is None or isinstance(result, UncertainArray)


@given(uncertain_arrays(length=12))
def test_check_and_load_cache_returns_something_for_existing_file(
    uncertain_array: UncertainArray,
) -> None:
    _store_cache(uncertain_array)
    assert _check_and_load_cache(12) is not None


def test_store_cache_expects_parameter_uncertain_values_as_uncertain_array() -> None:
    assert (
        signature(_store_cache).parameters["uncertain_values"].annotation
        is UncertainArray
    )


def test_cache_path_expects_stats_to_return_path() -> None:
    assert signature(_cache_path).return_annotation is Path


def test_dataset_extract_samples_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(provide_zema_samples).parameters


def test_dataset_extract_samples_expects_parameter_n_samples_as_int() -> None:
    assert signature(provide_zema_samples).parameters["n_samples"].annotation is int


def test_dataset_extract_samples_parameter_n_samples_default_is_one() -> None:
    assert signature(provide_zema_samples).parameters["n_samples"].default == 1


def test_dataset_extract_samples_states_to_return_uncertain_array() -> None:
    assert signature(provide_zema_samples).return_annotation is UncertainArray


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array(n_samples: int) -> None:
    assert isinstance(provide_zema_samples(n_samples), UncertainArray)


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array_with_n_samples_values(
    n_samples: int,
) -> None:
    assert len(provide_zema_samples(n_samples).values) == n_samples


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_array_with_n_samples_uncertainties(
    n_samples: int,
) -> None:
    result_uncertainties = provide_zema_samples(n_samples).uncertainties
    assert result_uncertainties is not None
    assert len(result_uncertainties) == n_samples


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_values_of_eleven_sensors(
    n_samples: int,
) -> None:
    assert provide_zema_samples(n_samples).values.shape[1] == 11


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_uncertainties_of_eleven_sensors(
    n_samples: int,
) -> None:
    result_uncertainties = provide_zema_samples(n_samples).uncertainties
    assert result_uncertainties is not None
    assert result_uncertainties.shape[1] == 11


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_values_and_uncertainties_which_are_not_similar(
    n_samples: int,
) -> None:
    result = provide_zema_samples(n_samples)
    assert not np.all(result.values == result.uncertainties)
