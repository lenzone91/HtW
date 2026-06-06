###############################
# Default AddNoiseAtSNR config
###############################

DEFAULT_ADD_NOISE_AT_SNR_CONFIG = {
    "input_key": "mixture",
    "output_key": "mixture",
    "snr_db": 20.0,
    "eps": 1e-8,
}


#######################################
# Default transform pipeline config
#######################################

DEFAULT_TRANSFORMS_CONFIG = {
    "add_noise_at_snr": dict(DEFAULT_ADD_NOISE_AT_SNR_CONFIG),
}

EMPTY_TRANSFORMS_CONFIG = {}


#######################################
# Default TSE waveform collator config
#######################################

DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIG = {
    "strict": True,
    "collator_config": {},
    "transforms": dict(EMPTY_TRANSFORMS_CONFIG),
}

DEFAULT_NOISY_TSE_WAVEFORM_COLLATOR_CONFIG = {
    "strict": True,
    "collator_config": {},
    "transforms": dict(DEFAULT_TRANSFORMS_CONFIG),
}


#######################################
# Default named collator configs
#######################################

DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS = {
    "tse_waveform": dict(DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIG),
}

DEFAULT_NOISY_TSE_WAVEFORM_COLLATOR_CONFIGS = {
    "tse_waveform": dict(DEFAULT_NOISY_TSE_WAVEFORM_COLLATOR_CONFIG),
}