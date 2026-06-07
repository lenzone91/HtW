
def build_ac_video_jepa(config):
    encoder =  build_block(config['encoder'])
    aencoder =  build_block(config['aencoder'])
    predictor =  build_block(config['predictor'])
    regularizer =  build_block(config['regularizer'])
    predcost =  build_block(config['predcost'])

def build_block(config):
    model_class = find_model_class(config.pop('model_class'))
    return model_class(**config)


def find_model_class():
    pass
