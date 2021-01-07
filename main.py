import os

import hydra

import pytorch_lightning as pl
from MyData import MyData
from MyModel import MyModel


@hydra.main(config_path="configs/", config_name="config.yaml")
def my_app(hparams):
    os.chdir(hydra.utils.get_original_cwd())
    model = MyModel(hparams.model)
    data = MyData(hparams.data)

    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    my_app()
