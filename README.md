# galaxyHackers

## Model training
![alt text](/screenshots/screenshot0.png)

## Clone the repo
```
git clone https://github.com/pelancha/galaxyHackers/
```

## Installing Dependencies (UNIX)

We assume that python^3.10 is installed on your system and poetry is installed on this python.

Create virtual environment with this python and activate it.
```
cd galaxyHackers
python3.10 -m venv venv
source ./venv/bin/activate
```

Install dependencies with poetry
```
poetry shell
poetry install
```

Caught a strange error (MacOS Sonoma). Installing library pixell through poetry is failing.
Solution:

poetry shell
pip install pixell=={version from pyproject.toml}
poetry install

## Usage

### Unpacking existing dataset

unzip dataset.zip -d ./storage/data
<!-- ```
cd galaxyHackers/models
python3 main.py --model MODEL_NAME --epoch NUM_EPOCH --lr LR
```
It is possible to run a script with several training model at the same time.
Pay attention that for chosen models will be applied the same optimizer with the same learning rate. Choice of several optimizers is not provided. -->
