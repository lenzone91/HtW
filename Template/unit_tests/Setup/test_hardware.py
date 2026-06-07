"""
Tests for hardware setup utilities.

This file validates hardware diagnostics and hardware request checks without
depending on the local machine having a CUDA-compatible GPU.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.hardware import (
    check_accelerator_request,
    check_cuda_request,
    check_requested_devices,
    get_cuda_device_names,
    get_hardware_info,
    setup_hardware,
)


#############################################
# Hardware information
#############################################


def test_get_hardware_info_returns_expected_keys() -> None:
    hardware_info = get_hardware_info()

    assert set(hardware_info.keys()) == {
        "cuda_available",
        "cuda_device_count",
        "cuda_device_names",
    }


def test_get_cuda_device_names_returns_empty_list_for_zero_devices() -> None:
    cuda_device_names = get_cuda_device_names(cuda_device_count=0)

    assert cuda_device_names == []


#############################################
# CUDA request checks
#############################################


def test_check_cuda_request_accepts_when_cuda_not_required() -> None:
    hardware_info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    check_cuda_request(
        require_cuda=False,
        hardware_info=hardware_info,
    )


def test_check_cuda_request_rejects_required_cuda_when_unavailable() -> None:
    hardware_info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    with pytest.raises(RuntimeError):
        check_cuda_request(
            require_cuda=True,
            hardware_info=hardware_info,
        )


#############################################
# Device request checks
#############################################


def test_check_requested_devices_accepts_none() -> None:
    hardware_info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    check_requested_devices(
        devices=None,
        hardware_info=hardware_info,
    )


def test_check_requested_devices_accepts_non_integer_lightning_semantics() -> None:
    hardware_info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    check_requested_devices(
        devices="auto",
        hardware_info=hardware_info,
    )


def test_check_requested_devices_rejects_too_many_devices() -> None:
    hardware_info = {
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_names": ["dummy_cuda_device"],
    }

    with pytest.raises(RuntimeError):
        check_requested_devices(
            devices=2,
            hardware_info=hardware_info,
            strict=True,
        )


#############################################
# Accelerator request checks
#############################################


def test_check_accelerator_request_accepts_cpu() -> None:
    hardware_info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    check_accelerator_request(
        accelerator="cpu",
        hardware_info=hardware_info,
    )


def test_check_accelerator_request_rejects_invalid_accelerator() -> None:
    hardware_info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    with pytest.raises(RuntimeError):
        check_accelerator_request(
            accelerator="invalid_accelerator",
            hardware_info=hardware_info,
            strict=True,
        )


#############################################
# Hardware setup
#############################################


def test_setup_hardware_returns_requested_and_available_context() -> None:
    context = setup_hardware(
        hardware_config={
            "strict": True,
            "accelerator": "cpu",
            "devices": None,
            "require_cuda": False,
        },
    )

    assert set(context.keys()) == {"requested", "available"}
    assert context["requested"] == {
        "accelerator": "cpu",
        "devices": None,
        "require_cuda": False,
    }
    assert set(context["available"].keys()) == {
        "cuda_available",
        "cuda_device_count",
        "cuda_device_names",
    }