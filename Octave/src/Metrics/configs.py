###########################################
###########################################
###########################################
# Objective metrics default configs
###########################################
###########################################
###########################################



########################
# Default snr config
########################

DEFAULT_SNR_CONFIG = {
    "zero_mean": False,
    "reduction": "mean",
}


##########################
# Default SI-SDR config
##########################

DEFAULT_SISDR_CONFIG = {
    "zero_mean": False,
    "reduction": "mean",
}

#########################
# Default SI-SIR
#########################


DEFAULT_SISIR_CONFIG = {
    "eps": 1e-8,
    "reduction": "mean",
}

#########################
# Default SI-SAR
#########################


DEFAULT_SISAR_CONFIG = {
    "eps": 1e-8,
    "reduction": "mean",
}


#########################
# Default SD-SDR config
#########################

DEFAULT_SDSDR_CONFIG = {
    "mu": 1.0,
    "eps": 1e-8,
    "reduction": "mean",
}



##########################
# Default DTW config
###########################


DEFAULT_DTW_CONFIG = {
    "reduction": "mean",
    "use_pruning": True,
}


############################
# Default LP norm config
############################

DEFAULT_LP_NORM_CONFIG = {
    "p": 2.0,
    "reduction": "mean",
}

############################
# Default LP error config
############################

DEFAULT_LP_ERROR_CONFIG = {
    "p": 2.0,
    "reduction": "mean",
}


###########################
# Default LSD config
###########################

DEFAULT_LSD_CONFIG = {
    "n_fft": 512,
    "hop_length": None,
    "win_length": None,
    "window": "hann",
    "center": True,
    "normalized": False,
    "onesided": True,
    "eps": 1e-8,
    "reduction": "mean",
}


################################
# Default Spectral KL config
################################

DEFAULT_SPECTRAL_KL_CONFIG = {
    "n_fft": 512,
    "hop_length": None,
    "win_length": None,
    "window": "hann",
    "center": True,
    "normalized": False,
    "onesided": True,
    "eps": 1e-8,
    "reduction": "mean",
     "normalize": True
}


#################################
# Default Itakura-Saito config
#################################

DEFAULT_ITAKURA_SAITO_CONFIG = {
    "n_fft": 512,
    "hop_length": None,
    "win_length": None,
    "window": "hann",
    "center": True,
    "normalized": False,
    "onesided": True,
    "eps": 1e-8,
    "reduction": "mean",
}



###########################################
###########################################
###########################################
# Subjective metrics default configs
###########################################
###########################################
###########################################

########################
# Default STOI
########################

DEFAULT_STOI_CONFIG = {
    "sample_rate": 16000,
    "reduction": "mean",
}


###########################
# Default ESTOI config
###########################

DEFAULT_ESTOI_CONFIG = {
    "sample_rate": 16000,
    "reduction": "mean",
}

###########################
# Default PESQ config
###########################

DEFAULT_PESQ_CONFIG = {
    "sample_rate": 16000,
    "mode": "wb",
    "reduction": "mean",
}

#############################
# Default DNSMOS P.808 config
#############################

DEFAULT_DNSMOSP808_CONFIG = {
    "sample_rate": 16000,
    "personalized": False,
    "device": None,
    "num_threads": None,
    "cache_session": True,
    "reduction": "mean",
}

#############################
# Default DNSMOS P.835 config
#############################

DEFAULT_DNSMOSP835_CONFIG = {
    "sample_rate": 16000,
    "personalized": False,
    "device": None,
    "num_threads": None,
    "cache_session": True,
    "reduction": "mean",
}

#############################
# Default DNSMOS SIG config
#############################

DEFAULT_DNSMOS_SIG_CONFIG = {
    "sample_rate": 16000,
    "personalized": False,
    "device": None,
    "num_threads": None,
    "cache_session": True,
    "reduction": "mean",
}

#############################
# Default DNSMOS BAK config
#############################

DEFAULT_DNSMOS_BAK_CONFIG = {
    "sample_rate": 16000,
    "personalized": False,
    "device": None,
    "num_threads": None,
    "cache_session": True,
    "reduction": "mean",
}

#############################
# Default DNSMOS OVRL config
#############################

DEFAULT_DNSMOS_OVRL_CONFIG = {
    "sample_rate": 16000,
    "personalized": False,
    "device": None,
    "num_threads": None,
    "cache_session": True,
    "reduction": "mean",
}


#########################
#########################
#########################
# Default metric config
#########################
#########################
#########################


######################
# All metrics
######################


DEFAULT_TSE_METRIC_SET_CONFIG = {
    "set_type": "tse",
    "strict": True,
    "metric_to_input_names": None,
    "metrics": {
        # Objective intrusive metrics.
        "snr": dict(DEFAULT_SNR_CONFIG),
        "sisdr": dict(DEFAULT_SISDR_CONFIG),
        "sdsdr": dict(DEFAULT_SDSDR_CONFIG),
        "dtw": dict(DEFAULT_DTW_CONFIG),
        "lp_error": dict(DEFAULT_LP_ERROR_CONFIG),

        # Scale-invariant decomposition metrics.
        "sisir": dict(DEFAULT_SISIR_CONFIG),
        "sisar": dict(DEFAULT_SISAR_CONFIG),

        # Spectral intrusive metrics.
        "lsd": dict(DEFAULT_LSD_CONFIG),
        "spectral_kl": dict(DEFAULT_SPECTRAL_KL_CONFIG),
        "itakura_saito": dict(DEFAULT_ITAKURA_SAITO_CONFIG),

        # Perceptual intrusive metrics.
        "stoi": dict(DEFAULT_STOI_CONFIG),
        "estoi": dict(DEFAULT_ESTOI_CONFIG),
        "pesq": dict(DEFAULT_PESQ_CONFIG),

        # Non-intrusive perceptual metrics.
        "dnsmos_p808": dict(DEFAULT_DNSMOSP808_CONFIG),
        "dnsmos_p835": dict(DEFAULT_DNSMOSP835_CONFIG),

        # Already included in dnsmosp835
        "dnsmos_sig": dict(DEFAULT_DNSMOS_SIG_CONFIG),
        "dnsmos_bak": dict(DEFAULT_DNSMOS_BAK_CONFIG),
        "dnsmos_ovrl": dict(DEFAULT_DNSMOS_OVRL_CONFIG),
    },
}


###########################
# All non redundant metrics
###########################


DEFAULT_NO_REDUNDANT_TSE_METRIC_SET_CONFIG = {
    "set_type": "tse",
    "strict": True,
    "metric_to_input_names": None,
    "metrics": {
        # Objective intrusive metrics.
        "snr": dict(DEFAULT_SNR_CONFIG),
        "sisdr": dict(DEFAULT_SISDR_CONFIG),
        "sdsdr": dict(DEFAULT_SDSDR_CONFIG),
        "dtw": dict(DEFAULT_DTW_CONFIG),
        "lp_error": dict(DEFAULT_LP_ERROR_CONFIG),

        # Scale-invariant decomposition metrics.
        "sisir": dict(DEFAULT_SISIR_CONFIG),
        "sisar": dict(DEFAULT_SISAR_CONFIG),

        # Spectral intrusive metrics.
        "lsd": dict(DEFAULT_LSD_CONFIG),
        "spectral_kl": dict(DEFAULT_SPECTRAL_KL_CONFIG),
        "itakura_saito": dict(DEFAULT_ITAKURA_SAITO_CONFIG),

        # Perceptual intrusive metrics.
        "stoi": dict(DEFAULT_STOI_CONFIG),
        "estoi": dict(DEFAULT_ESTOI_CONFIG),
        "pesq": dict(DEFAULT_PESQ_CONFIG),

        # Non-intrusive perceptual metrics.
        "dnsmos_p808": dict(DEFAULT_DNSMOSP808_CONFIG),
        "dnsmos_p835": dict(DEFAULT_DNSMOSP835_CONFIG),

    },
}

########################
# All non heavy metrics
########################


DEFAULT_TRAIN_TSE_METRIC_SET_CONFIG = {
    "set_type": "tse",
    "strict": True,
    "metric_to_input_names": None,
    "metrics": {
        # Objective intrusive metrics.
        "snr": dict(DEFAULT_SNR_CONFIG),
        "sisdr": dict(DEFAULT_SISDR_CONFIG),
        "sdsdr": dict(DEFAULT_SDSDR_CONFIG),
        "lp_error": dict(DEFAULT_LP_ERROR_CONFIG),

        # Scale-invariant decomposition metrics.
        "sisir": dict(DEFAULT_SISIR_CONFIG),
        "sisar": dict(DEFAULT_SISAR_CONFIG),

        # Spectral intrusive metrics.
        "lsd": dict(DEFAULT_LSD_CONFIG),
        "spectral_kl": dict(DEFAULT_SPECTRAL_KL_CONFIG),
        "itakura_saito": dict(DEFAULT_ITAKURA_SAITO_CONFIG),
    },
}




#########################
# Default loss config
#########################

DEFAULT_TSE_LOSS_CONFIG = {
    "loss_type": "weighted_metric",
    "name": "loss",
    "strict": True,
    "return_loss_terms": True,
    "metric_weights": {
        # Differentiable metrics to maximize.
        "snr": -1.0,
        "sisdr": -1.0,
        "sdsdr": -1.0,
        "sisir": -1.0,
        "sisar": -1.0,

        # Differentiable metrics to minimize.
        "lp_error": 1.0,
        "lsd": 1.0,
        "spectral_kl": 1.0,
        "itakura_saito": 1.0,

        # Evaluation-only / non-differentiable metrics.
        "dtw": None,
        "stoi": None,
        "estoi": None,
        "pesq": None,
        "dnsmos_p808": None,
        "dnsmos_p835": None,
    },
}