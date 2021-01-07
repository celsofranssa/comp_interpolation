# Comp Interpolation Challenge

### 1. Quick Start

```shell script
# clone the project 
git git@github.com:celsofranssa/comp_interpolation.git

# change directory to project folder
cd comp_interpolation/

# Create a new virtual environment by choosing a Python interpreter 
# and making a ./venv directory to hold it:
virtualenv -p python3 ./venv

# activate the virtual environment using a shell-specific command:
source ./venv/bin/activate

# install dependecies
pip install -r requirements.txt

# setting python path (if you need)
export PYTHONPATH=$PATHONPATH:<path-to-project-dir>/comp_interpolation/

# (if you need) to exit virtualenv later:
deactivate
```

### 2. Test Run
The following bash command save the configs to a saved.config.yaml with the correct interpolation
```
python main.py model=gpt
```
The following output should be produced a `saved.config.yaml` file with the following content:
```
name: gpt
predictions:
  path: ../resources/predictions/gpt_dataset_02_predictions.pt
```
However, the following error will also happen:

```
Traceback (most recent call last):
  File "main.py", line 17, in my_app
    trainer.fit(model)
  File "/home/celso/projects/comp_interpolation/venv/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 472, in fit
    results = self.accelerator_backend.train()
  File "/home/celso/projects/comp_interpolation/venv/lib/python3.7/site-packages/pytorch_lightning/accelerators/cpu_accelerator.py", line 59, in train
    self.trainer.train_loop.setup_training(model)
  File "/home/celso/projects/comp_interpolation/venv/lib/python3.7/site-packages/pytorch_lightning/trainer/training_loop.py", line 160, in setup_training
    self.trainer.logger.save()
  File "/home/celso/projects/comp_interpolation/venv/lib/python3.7/site-packages/pytorch_lightning/utilities/distributed.py", line 39, in wrapped_fn
    return fn(*args, **kwargs)
  File "/home/celso/projects/comp_interpolation/venv/lib/python3.7/site-packages/pytorch_lightning/loggers/tensorboard.py", line 221, in save
    save_hparams_to_yaml(hparams_file, self.hparams)
  File "/home/celso/projects/comp_interpolation/venv/lib/python3.7/site-packages/pytorch_lightning/core/saving.py", line 366, in save_hparams_to_yaml
    OmegaConf.save(hparams, fp, resolve=True)
omegaconf.errors.ConfigKeyError: str interpolation key 'data.name' not found

```
