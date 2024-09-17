# galaxyHackers

## Model training
![alt text](/screenshots/screenshot0.png)

## Clone the Repository

Since the repository is private, use the following command syntax to clone it:

```bash
git clone https://YOUR_CLASSIC_TOKEN@github.com/pelancha/galaxyHackers.git
```

Replace `YOUR_CLASSIC_TOKEN` with your GitHub personal access token.

## Installating Dependencies (UNIX)

We assume that Python 3.10+ is installed on your system, and you have Poetry installed for dependency management.

### Set Up Virtual Environment

1. Navigate to the project directory:
    ```bash
    cd galaxyHackers
    ```

2. Create a virtual environment using Python 3.10 and activate it:
    ```bash
    python3.10 -m venv venv
    source ./venv/bin/activate
    ```

### Install Dependencies with Poetry

1. Enter the Poetry environment:
    ```bash
    poetry shell
    ```

2. Install the project dependencies:
    ```bash
    poetry install
    ```

### **MacOS Sonoma**: Fixing `pixell` Installation Error

On MacOS Sonoma, you may encounter issues when installing `pixell` through Poetry. If this happens, follow these steps:

1. Enter the Poetry shell:
    ```bash
    poetry shell
    ```

2. Manually install the `pixell` library:
    ```bash
    pip install pixell=={version_from_pyproject.toml}
    ```

3. After installing `pixell`, re-run the Poetry installation:
    ```bash
    poetry install
    ```

### **Deprecated Method**: Using `pip` for Dependencies

If Poetry doesn't suit your needs, you can use `pip` to install all dependencies.

1. (Optional) Set up and activate the virtual environment as described above:
    ```bash
    python3.10 -m venv venv
    source ./venv/bin/activate
    ```

2. Install the necessary packages using `pip`:
    ```bash
    pip install torch torchvision timm torch_optimizer tqdm
    pip install numpy pandas matplotlib scikit-learn
    pip install Pillow astropy astroquery pixell
    pip install dynaconf wget comet_ml
    ```

## Dataset Preparation

Before training the models, you'll need to unpack the dataset:

Unzip the dataset into the `./storage/` directory:
```bash
unzip data.zip -d ./storage/
```

## Training Models

Once dependencies are installed and the dataset is unpacked, you can start training the models.

1. Navigate to the `models` directory:
    ```bash
    cd galaxyHackers/models
    ```

2. Train a model by running the `main.py` script:
    ```bash
    python3 main.py --model MODEL_NAME --epoch NUM_EPOCH --lr LR
    ```

    - `MODEL_NAME`: The name of the model you wish to train.
    - `NUM_EPOCH`: The number of epochs for training.
    - `LR`: Learning rate for the optimizer.

### Running Multiple Models Simultaneously

You can run several models simultaneously with the same optimizer and learning rate. However, please note that the script does not support using different optimizers for each model at this time.

---

### Example Commands

**Single model training**:
```bash
python3 main.py --model ResNet18 --epoch 50 --lr 0.001
```

**Multiple models training**:
```bash
python3 main.py --model AlexNet_VGG ResNet18 --epoch 50 --lr 0.001
```
```bash
python3 main.py --epoch 50 --lr 0.001
```