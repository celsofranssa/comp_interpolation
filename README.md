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
If all goes well the following output should be produced a `saved.config.yaml` file with the following content:
```
name: gpt
predictions:
  path: ../resources/predictions/gpt_dataset_02_predictions.pt
```