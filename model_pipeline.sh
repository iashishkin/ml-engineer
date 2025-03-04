echo "creating virtual environment ..."
python3 -m venv .env

source .env/bin/activate
echo "virtual environment activated"

echo "setting virtual environment ..."
pip instal -r requirements.txt

# preprocess data
echo "preprocessing train data ..."
python3 -m scripts.data_preprocessing --config="scripts/configs/train_sample_preprocessing.yaml"

echo "preprocessing val data ..."
python3 -m scripts.data_preprocessing --config="scripts/configs/val_sample_preprocessing.yaml"

echo "preprocessing test data ..."
python3 -m scripts.data_preprocessing --config="scripts/configs/test_sample_preprocessing.yaml"

# train model
echo "train model ..."
python3 -m scripts.model_train --config="scripts/configs/train_config.yaml"


# inference model
echo "inference trained model on train data ..."
python3 -m scripts.model_inference --config="scripts/configs/inference_train.yaml"

echo "inference trained model on val data ..."
python3 -m scripts.model_inference --config="scripts/configs/inference_val.yaml"

echo "inference trained model on test data ..."
python3 -m scripts.model_inference --config="scripts/configs/inference_test.yaml"