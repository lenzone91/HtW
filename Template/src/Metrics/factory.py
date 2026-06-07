from torch import nn
from tqdm import tqdm

from .objective import (
    SNR,
    SISDR,
    SISIR,
    SISAR,
    SDSDR,
    DTW,
    LPNorm,
    LPError,
    LSD,
    SpectralKLDivergence,
    ItakuraSaitoDivergence,
)

from .subjective import (
    STOI,
    ESTOI,
    PESQ,
    DNSMOSP808,
    DNSMOSP835,
    DNSMOSSIG,
    DNSMOSBAK,
    DNSMOSOVRL,
)

from .MetricSet import MetricSet, LoggableMetricSet, TSEMetricSet
from .loss import WeightedMetricLoss
from ..Factory.base import BaseBuilder, BaseBuilderDispatcher

from .configs import (
    DEFAULT_SNR_CONFIG,
    DEFAULT_SISDR_CONFIG,
    DEFAULT_SISIR_CONFIG,
    DEFAULT_SISAR_CONFIG,
    DEFAULT_SDSDR_CONFIG,
    DEFAULT_DTW_CONFIG,
    DEFAULT_LP_ERROR_CONFIG,
    DEFAULT_LSD_CONFIG,
    DEFAULT_SPECTRAL_KL_CONFIG,
    DEFAULT_ITAKURA_SAITO_CONFIG,
    DEFAULT_STOI_CONFIG,
    DEFAULT_ESTOI_CONFIG,
    DEFAULT_PESQ_CONFIG,
    DEFAULT_DNSMOSP808_CONFIG,
    DEFAULT_DNSMOSP835_CONFIG,
    DEFAULT_DNSMOS_SIG_CONFIG,
    DEFAULT_DNSMOS_BAK_CONFIG,
    DEFAULT_DNSMOS_OVRL_CONFIG,
    DEFAULT_TSE_METRIC_SET_CONFIG,
    DEFAULT_TSE_LOSS_CONFIG,
)


###########################################################
# MetricBuilder
###########################################################


class MetricBuilder(BaseBuilder):
    """
    Build one metric instance from its keyword config.
    """

    def __init__(
        self,
        metric_class: type[nn.Module],
        default_config: dict,
        strict: bool = True,
        check_default_keys: bool = True,
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

        self.metric_class = metric_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> nn.Module:
        return self.metric_class(**config)


##################################
# MetricBuilder registry
##################################


METRIC_BUILDERS_REGISTRY = {
    "snr": MetricBuilder(SNR, DEFAULT_SNR_CONFIG),
    "sisdr": MetricBuilder(SISDR, DEFAULT_SISDR_CONFIG),
    "sisir": MetricBuilder(SISIR, DEFAULT_SISIR_CONFIG),
    "sisar": MetricBuilder(SISAR, DEFAULT_SISAR_CONFIG),
    "sdsdr": MetricBuilder(SDSDR, DEFAULT_SDSDR_CONFIG),
    "dtw": MetricBuilder(DTW, DEFAULT_DTW_CONFIG),
    "lp_error": MetricBuilder(LPError, DEFAULT_LP_ERROR_CONFIG),

    "lsd": MetricBuilder(LSD, DEFAULT_LSD_CONFIG),
    "spectral_kl": MetricBuilder(SpectralKLDivergence, DEFAULT_SPECTRAL_KL_CONFIG),
    "itakura_saito": MetricBuilder(ItakuraSaitoDivergence, DEFAULT_ITAKURA_SAITO_CONFIG),

    "stoi": MetricBuilder(STOI, DEFAULT_STOI_CONFIG),
    "estoi": MetricBuilder(ESTOI, DEFAULT_ESTOI_CONFIG),
    "pesq": MetricBuilder(PESQ, DEFAULT_PESQ_CONFIG),

    "dnsmos_p808": MetricBuilder(DNSMOSP808, DEFAULT_DNSMOSP808_CONFIG),
    "dnsmos_p835": MetricBuilder(DNSMOSP835, DEFAULT_DNSMOSP835_CONFIG),
    "dnsmos_sig": MetricBuilder(DNSMOSSIG, DEFAULT_DNSMOS_SIG_CONFIG),
    "dnsmos_bak": MetricBuilder(DNSMOSBAK, DEFAULT_DNSMOS_BAK_CONFIG),
    "dnsmos_ovrl": MetricBuilder(DNSMOSOVRL, DEFAULT_DNSMOS_OVRL_CONFIG),
}


##############################################
# MetricDispatcher
##############################################


class MetricDispatcher(BaseBuilderDispatcher):
    """
    Build several metrics from a metric config dictionary.
    """

    def __init__(
        self,
        builder_registry: dict = METRIC_BUILDERS_REGISTRY,
        strict: bool = True,
        show_progress: bool = False,
    ):
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

        self.show_progress = show_progress

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> dict[str, nn.Module]:
        metrics = {}

        iterator = tqdm(
            config.items(),
            desc="Building metrics",
            disable=not self.show_progress,
        )

        for metric_name, metric_config in iterator:
            metric = self.build_one(
                object_name=metric_name,
                object_config=metric_config,
                runtime_context=runtime_context,
            )

            if metric is not None:
                metrics[metric_name] = metric

        return metrics

    def handle_unknown_object(self, object_name: str) -> None:
        self.handle_error(
            f"Unknown metric '{object_name}'. "
            f"Available metrics are: {sorted(self.builder_registry.keys())}."
        )


###########################################
# MetricSetBuilder
###########################################


class MetricSetBuilder(BaseBuilder):
    """
    Build MetricSet instances from metric configs.
    """

    def __init__(
        self,
        default_config: dict,
        metric_dispatcher: MetricDispatcher | None = None,
        strict: bool = True,
        show_progress: bool = False,
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=False,
        )

        if metric_dispatcher is None:
            metric_dispatcher = MetricDispatcher(
                strict=strict,
                show_progress=show_progress,
            )

        self.metric_dispatcher = metric_dispatcher

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> MetricSet:
        metrics = self.metric_dispatcher(
            config=config["metrics"],
            runtime_context=runtime_context,
        )

        return MetricSet(
            strict=config["strict"],
            **metrics,
        )


#######################################
# LoggableMetricSetBuilder
#######################################


class LoggableMetricSetBuilder(MetricSetBuilder):
    """
    Build LoggableMetricSet instances from metric configs.
    """

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> LoggableMetricSet:
        metrics = self.metric_dispatcher(
            config=config["metrics"],
            runtime_context=runtime_context,
        )

        return LoggableMetricSet(
            strict=config["strict"],
            **metrics,
        )


###########################################
# TSEMetricSetBuilder
###########################################


class TSEMetricSetBuilder(LoggableMetricSetBuilder):
    """
    Build TSEMetricSet instances from metric configs.
    """

    def __init__(
        self,
        metric_dispatcher: MetricDispatcher | None = None,
        strict: bool = True,
        show_progress: bool = False,
    ):
        super().__init__(
            default_config=DEFAULT_TSE_METRIC_SET_CONFIG,
            metric_dispatcher=metric_dispatcher,
            strict=strict,
            show_progress=show_progress,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> TSEMetricSet:
        metrics = self.metric_dispatcher(
            config=config["metrics"],
            runtime_context=runtime_context,
        )

        return TSEMetricSet(
            strict=config["strict"],
            metric_to_input_names=config["metric_to_input_names"],
            **metrics,
        )


################################################
# LossBuilder
################################################


class LossBuilder(BaseBuilder):
    """
    Build loss objects from plain serializable configs.
    """

    def __init__(
        self,
        default_config: dict = DEFAULT_TSE_LOSS_CONFIG,
        strict: bool = True,
        check_default_keys: bool = True,
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> WeightedMetricLoss:
        loss_type = config["loss_type"]

        match loss_type:
            case "weighted_metric":
                return WeightedMetricLoss(
                    metric_weights=config["metric_weights"],
                    strict=config["strict"],
                    name=config["name"],
                    return_loss_terms=config["return_loss_terms"],
                )

            case _:
                self.handle_error(
                    f"Unknown loss type: {loss_type}. "
                    "Expected one of {'weighted_metric'}."
                )
                return None


###################################
# build_metric_set
###################################


def build_metric_set(
    config: dict,
    runtime_context: dict | None = None,
    show_progress: bool = False,
):
    """
    Build one metric set from a metric set config.
    """
    set_type = config.get("set_type", "tse")
    strict = config.get("strict", True)

    match set_type:
        case "metric":
            builder = MetricSetBuilder(
                default_config=DEFAULT_TSE_METRIC_SET_CONFIG,
                strict=strict,
                show_progress=show_progress,
            )

        case "loggable":
            builder = LoggableMetricSetBuilder(
                default_config=DEFAULT_TSE_METRIC_SET_CONFIG,
                strict=strict,
                show_progress=show_progress,
            )

        case "tse":
            builder = TSEMetricSetBuilder(
                strict=strict,
                show_progress=show_progress,
            )

        case _:
            raise ValueError(
                f"Unknown metric set type: {set_type}. "
                "Expected one of {'metric', 'loggable', 'tse'}."
            )

    return builder(
        config=config,
        runtime_context=runtime_context,
    )


####################################
# build_loss
####################################


def build_loss(
    config: dict,
    runtime_context: dict | None = None,
) -> WeightedMetricLoss:
    builder = LossBuilder(
        strict=config.get("strict", True),
    )

    return builder(
        config=config,
        runtime_context=runtime_context,
    )
