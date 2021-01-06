import os
import tempfile

import hydra
from omegaconf import OmegaConf


class MyModel:
    def __init__(self, hparams):
        with open("saved.config.yaml", "w", encoding="utf-8") as fp:
            OmegaConf.save(hparams, fp, resolve=True)



class MyData:
    def __init__(self, hparams):
        print(hparams)


@hydra.main(config_path="configs/", config_name="config.yaml")
def my_app(hparams):
    os.chdir(hydra.utils.get_original_cwd())
    model = MyModel(hparams.model)
    data = MyData(hparams.data)


if __name__ == '__main__':
    my_app()
