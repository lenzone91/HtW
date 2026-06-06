from .base import BaseLightningModule

from ..Optimization.factory import build_optimizer, build_scheduler

from eb_jepa.jepa import JEPA


class AcVideoJepaModule(BaseLightningModule):
    def __init__(self, 
                 encoder, 
                 aencoder, 
                 predictor, 
                 regularizer, 
                 predcost, 
                 optimizer_config: dict, 
                 scheduler_config: dict = {}, 
                 strict: bool = True):
        
        super().__init__(strict)
        self.encoder = encoder
        self.aencoder = aencoder
        self.predictor = predictor
        self.regularizer = regularizer
        self.predcost = predcost
        self.jepa = JEPA(encoder, aencoder, predictor, regularizer, predcost)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config







    def training_step(self, batch, batch_idx: int):
        x, a, loc, _, _ = batch

        _, (jepa_loss, regl, regl_unweight, regldict, pl) = self.jepa.unroll(
                    x,
                    a,
                    nsteps=cfg.model.nsteps,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
        
        self.log_step_dict({'jepa_loss' : jepa_loss},
                           ml_step = 'train',
                           )
        return jepa_loss
    
    def test_step(self, batch, batch_idx: int):
        return super().test_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = build_optimizer(
            parameters=self.jepa.parameters(),
            optimizer_config=self.optimizer_config,
            strict=self.strict
            )
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_config=self.scheduler_config,
            strict=self.strict
            )

        

        if len(scheduler) == 0:
            return optimizer

        return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }