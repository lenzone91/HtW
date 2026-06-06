from pathlib import Path

import torch

from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from .toy_dataset import SourceSeparationDataset
from .configs import DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG


############################
# Builder
############################


class DatasetBuilder(BaseBuilder):
    """
    Build one dataset from a serializable dataset config.

    The builder may use runtime_context to resolve runtime-dependent paths.

    Dataset constructors receive resolved paths only.
    """

    def __init__(
        self,
        dataset_class: type,
        default_config: dict,
        strict: bool = True,
        check_default_keys: bool = True,
    ) -> None:
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

        self.dataset_class = dataset_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ):
        dataset_kwargs = self.build_dataset_kwargs(
            config=config,
            runtime_context=runtime_context,
        )

        return self.dataset_class(**dataset_kwargs)

    def build_dataset_kwargs(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> dict:
        """
        Convert a serializable config into dataset constructor kwargs.
        """
        dataset_kwargs = dict(config)

        dataset_kwargs["path_to_folder"] = self.resolve_dataset_path(
            config=config,
            runtime_context=runtime_context,
        )

        dataset_kwargs["canonical_dtype"] = self.resolve_dtype(
            dtype_name=config["canonical_dtype"],
        )

        dataset_kwargs.pop("path", None)
        dataset_kwargs.pop("path_key", None)

        return dataset_kwargs

    def resolve_dataset_path(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> str:
        """
        Resolve the dataset path from either:
            - config["path"], directly;
            - config["path_key"], through runtime_context["data"]["dataset_roots"].
        """
        path = config.get("path", None)
        path_key = config.get("path_key", None)

        if path is not None and path_key is not None:
            self.handle_error(
                "Dataset config cannot define both 'path' and 'path_key'."
            )
            return ""

        if path is not None:
            resolved_path = Path(path).expanduser().resolve()
            return str(resolved_path)

        if path_key is not None:
            return self.resolve_dataset_path_from_key(
                path_key=path_key,
                runtime_context=runtime_context,
            )

        self.handle_error("Dataset config must define either 'path' or 'path_key'.")
        return ""

    def resolve_dataset_path_from_key(
        self,
        path_key: str,
        runtime_context: dict | None = None,
    ) -> str:
        """
        Resolve a dataset path key through runtime_context["data"]["dataset_roots"].
        """
        if runtime_context is None:
            self.handle_error(
                "runtime_context is required when dataset config uses 'path_key'."
            )
            return ""

        if "data" not in runtime_context:
            self.handle_error("runtime_context does not contain a 'data' entry.")
            return ""

        data_context = runtime_context["data"]

        if "dataset_roots" not in data_context:
            self.handle_error(
                "runtime_context['data'] does not contain a 'dataset_roots' entry."
            )
            return ""

        dataset_roots = data_context["dataset_roots"]

        if path_key not in dataset_roots:
            self.handle_error(
                f"Unknown dataset path_key '{path_key}'. "
                f"Available dataset roots are: {sorted(dataset_roots.keys())}."
            )
            return ""

        resolved_path = Path(dataset_roots[path_key]).expanduser().resolve()
        return str(resolved_path)

    @staticmethod
    def resolve_dtype(
        dtype_name: str,
    ) -> torch.dtype:
        """
        Convert a serializable dtype name into a torch dtype.
        """
        dtype_registry = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype_name not in dtype_registry:
            raise ValueError(
                f"Unknown dtype '{dtype_name}'. "
                f"Available dtypes are: {sorted(dtype_registry.keys())}."
            )

        return dtype_registry[dtype_name]


#####################################
# Registry
#####################################


DATASET_BUILDERS_REGISTRY = {
    "source_separation": DatasetBuilder(
        dataset_class=SourceSeparationDataset,
        default_config=DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG,
    ),
}


#######################################
# Dispatcher
#######################################


class DatasetDispatcher(BaseBuilderDispatcher):
    """
    Build several datasets from a dataset config dictionary.

    Expected config:
        dataset_configs[dataset_name] = dataset_config
    """

    def __init__(
        self,
        builder_registry: dict = DATASET_BUILDERS_REGISTRY,
        strict: bool = True,
    ) -> None:
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )


###################################
# Wrapper
###################################


def build_datasets(
    dataset_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build datasets from a datasets config dictionary.
    """
    dispatcher = DatasetDispatcher(
        strict=dataset_configs.get("strict", True),
    )

    resolved_dataset_configs = dict(dataset_configs)
    resolved_dataset_configs.pop("strict", None)

    return dispatcher(
        config=resolved_dataset_configs,
        runtime_context=runtime_context,
    )