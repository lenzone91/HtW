from lightning.pytorch.callbacks import ModelCheckpoint

class NamedModelCheckpoint(ModelCheckpoint):
    def __init__(self, checkpoint_name: str, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_name = checkpoint_name

    @property
    def state_key(self) -> str:
        return f"{super().state_key}|checkpoint_name={self.checkpoint_name}"