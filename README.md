# ml-engineer
**ml-engineer** is a modular machine learning project designed for data preprocessing, model training, and inference. 
It includes Python scripts, a set of utilities, bash scripts for automating the pipeline, and Jupyter notebooks for 
exploratory data analysis (EDA). Each stage of the pipeline is fully controlled by corresponding configuration files 
(typically in YAML format), allowing flexible control over data filtering, feature transformations, and model parameters.


## Project struture

```
ml-engineer/
├── dataset/                  
│   └── (data files, raw and/or processed)
├── notebooks/                
│   └── eda.ipynb             # Jupyter notebook for EDA on the given data
├── scripts/                  
│   ├── data_preprocessing.py # Script for data preprocessing
│   ├── model_train.py        # Script for model training
│   ├── model_inference.py    # Script for model inference
│   ├── triain.py             # Deprecated model training script
│   └── configs/              # Configuration files controlling each major stage
├── src/                      
│   ├── data_preprocessing_utils.py  # Utilities for data preprocessing
│   ├── training_utils.py            # Utilities for model training and inference, including transform pipelines
│   ├── utils.py                     # General utilities (logging, config loading, etc.)
│   └── model.py                     # Deprecated dummy model implementation
├── model_pipeline.sh         # Bash script demonstrating the complete pipeline (virtual environment creation, data preprocessing, training, and inference)
└── requirements.txt          # Required libraries for setting up the virtual environment
```

*Note*: Deprecated files (e.g., train.py and model.py) are kept for reference but are no longer used in the main pipeline.


## Installation

1. **Clone the Repository:**

```commandline
git clone git@github.com:iashishkin/ml-engineer.git
cd ml-engineer
```

2. **Create and activate a virtual environment::**

```commandline
python3 -m venv .env
source .env/bin/activate
```

3. **Install the required packages using the provided requirements file:**

```commandline
pip install -r requirements.txt

```

## Usage

### Running the Entire Pipeline

A bash script (model_pipeline.sh) is provided to demonstrate the entire pipeline. This script handles:

    - Creating the virtual environment 
    - Running data preprocessing 
    - Training the model
    - Running model inference

To run the complete pipeline, simply execute:

```commandline
bash model_pipeline.sh
```

*Note*: Make sure you have configured the environment and configuration files as needed before running the script.

### Running Individual Stages

Each major stage of the project is fully configurable via YAML configuration files. 
The scripts below accept a configuration file as input:

1. **Data Preprocessing**

The data preprocessing stage is executed via the `data_preprocessing.py` script.

```commandline
python3 -m scripts.data_preprocessing --config="scripts/configs/your_preprocessing_config.yaml"
```

*The preprocessing config controls which features to use, how to handle missing data, duplicate removal,
and applies various filters.*

2. **Model Training**

Train your model using the `model_train.py` script:

```commandline
python3 -m scripts.model_train --config="scripts/configs/your_training_config.yaml"
```

*The training configuration file includes paths to training data, target column, feature transforms 
(e.g., combinations of filters and encoders), and advanced model parameters.*

3. **Model Inference**

Run inference using the `model_inference.py` script:

```commandline
python3 -m scripts.model_inference --config="scripts/configs/your_inference_config.yaml"
```

*The inference config specifies the model source, data source, and output destination for predictions. 
It also controls whether the target is loaded for loss computation.*

## Configuration Files

Each stage is fully controlled by its corresponding configuration file (found in the `scripts/configs directory`). 
For example:

- **Data Preprocessing Config:**

    Supports different combinations of filters applied to different features. 
    You can specify which columns to process, methods for handling missing values, duplicate removal parameters, 
    and filter rules (using regex patterns).


- **Model Training Config:**

    Controls the training data source, feature transformations, model hyperparameters, and options for early stopping if 
    validation data is provided.


- **Model Inference Config:**

    Specifies the paths for loading the saved feature transformation pipeline and model, as well as options to compute 
    prediction loss if target data is available.

Using these config files makes the pipeline highly flexible and easy to adjust without modifying the source code.

## Additional Information

- **Logging & Utilities:**

    The `src/utils.py` module provides essential logging functionality and configuration loading, 
    ensuring consistent logging across scripts.


- **Transform Pipelines:**

    The feature transformation pipeline is built using a global registry of transformers 
    (e.g., `LogTransform`, `DummyTransform`, and `CategoryDTypeTransform`).
    These are combined and applied to the data before training and inference.


- **Deprecated Files:**

    Some files, like `triain.py` and `model.py`, are deprecated. 
    They remain in the repository for historical context but are no longer used in the main pipeline.


- **Testing & Evaluation:**

    The `train.py` file provides functions to compute training loss and evaluate model performance
    (using metrics like MAPE). This file serves as an example for model evaluation.

[//]: # (Contributing)

[//]: # ()
[//]: # (Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.)

[//]: # (License)

[//]: # ()
[//]: # ([Specify your license here.])