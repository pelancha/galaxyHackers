# galaxyHackers

## Model training
![alt text](/screenshots/screenshot0.png)

## Clone the repo
```
git clone https://github.com/pelancha/galaxyHackers/
```

## Installing Dependencies
```
pip install -r requirements.txt
```
## Usage
```
cd galaxyHackers/models
python3 main.py --model MODEL_NAME --epoch NUM_EPOCH --lr LR
```
It is possible to run a script with several training model at the same time.
Pay attention that for chosen models will be applied the same optimizer with the same learning rate. Choice of several optimizers is not provided.
