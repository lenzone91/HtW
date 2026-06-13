from eb_jepa.jepa import JEPA


class AcVideoJepa(JEPA):
    """
    AcVideoJepa world model wrapper.
    """

    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        super().__init__(
            encoder=encoder,
            aencoder=aencoder,
            predictor=predictor,
            regularizer=regularizer,
            predcost=predcost,
        )
