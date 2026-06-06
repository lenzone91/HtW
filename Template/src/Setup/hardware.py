import torch

from ..Utils.error import handle_error

#############################################
# Hardware information
#############################################

# This file centralizes hardware-related setup checks.
#
# It does not create torch devices.
# It does not configure PyTorch Lightning Trainer.
# It only checks whether the requested hardware setup is compatible
# with the current runtime environment.


def get_hardware_info() -> dict:
    """
    Return basic information about the available hardware.
    """
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0

    return {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": get_cuda_device_names(cuda_device_count),
    }


def get_cuda_device_names(cuda_device_count: int) -> list[str]:
    """
    Return the names of all visible CUDA devices.
    """
    return [
        torch.cuda.get_device_name(device_index)
        for device_index in range(cuda_device_count)
    ]


#############################################
# Hardware checks
#############################################

# The setup config expresses what the user expects from the runtime.
# This file checks whether these expectations are compatible with the machine.


def check_cuda_request(
    require_cuda: bool,
    hardware_info: dict,
) -> None:
    """
    Check that CUDA is available when explicitly required.
    """
    if require_cuda and not hardware_info["cuda_available"]:
        raise RuntimeError(
            "CUDA was explicitly required, but no CUDA device is available."
        )


def check_requested_devices(
    devices: int | str | None,
    hardware_info: dict,
    strict: bool = True,
) -> None:
    """
    Check that the requested number of CUDA devices is available.

    This check only applies when devices is an integer.
    Other values are left to PyTorch Lightning semantics.
    """
    if devices is None:
        return

    if not isinstance(devices, int):
        return

    if devices <= 0:
        handle_error(
            msg=f"devices must be a positive integer, got {devices}.",
            strict=strict,
        )
        return

    available_devices = hardware_info["cuda_device_count"]

    if devices > available_devices:
        handle_error(
            msg=(
                f"{devices} CUDA devices were requested, "
                f"but only {available_devices} are available."
            ),
            strict=strict,
        )


def check_accelerator_request(
    accelerator: str | None,
    hardware_info: dict,
    strict: bool = True,
) -> None:
    """
    Check that the requested accelerator is compatible with the runtime.

    This function does not reproduce all Lightning accelerator semantics.
    It only catches clear impossible requests early.
    """
    if accelerator is None:
        return

    if accelerator == "cpu":
        return

    if accelerator == "cuda" and not hardware_info["cuda_available"]:
        handle_error(
            msg="CUDA accelerator was requested, but CUDA is not available.",
            strict=strict,
        )
        return

    if accelerator not in {"cpu", "cuda", "auto"}:
        handle_error(
            msg=(
                f"Unsupported accelerator value: {accelerator}. "
                "Expected one of {'cpu', 'cuda', 'auto'}."
            ),
            strict=strict,
        )


#############################################
# Hardware setup
#############################################

# setup_hardware is the public entry point of this file.
# It validates the hardware-related setup config and returns a hardware context.


def setup_hardware(hardware_config: dict | None = None) -> dict:
    """
    Validate hardware-related setup options and return hardware context.
    """
    if hardware_config is None:
        hardware_config = {}

    strict = hardware_config.get("strict", True)
    accelerator = hardware_config.get("accelerator", None)
    devices = hardware_config.get("devices", None)
    require_cuda = hardware_config.get("require_cuda", False)

    hardware_info = get_hardware_info()

    check_cuda_request(
        require_cuda=require_cuda,
        hardware_info=hardware_info,
    )

    check_accelerator_request(
        accelerator=accelerator,
        hardware_info=hardware_info,
        strict=strict,
    )

    check_requested_devices(
        devices=devices,
        hardware_info=hardware_info,
        strict=strict,
    )

    return {
        "requested": {
            "accelerator": accelerator,
            "devices": devices,
            "require_cuda": require_cuda,
        },
        "available": hardware_info,
    }


