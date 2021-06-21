# Using Language Model To Bootstrap Human Activity Recognition In Smart Homes

This repository is the official implementation of [Using Language Model To Bootstrap Human Activity Recognition In Smart Homes](https://arxiv.org/abs/2030.12345). 

## Requirements

To install requirements:

To use this repository you should download and install SmartHomeHARLib package

```setup
git clone git@github.com:dbouchabou/SmartHomeHARLib.git
pip install -r requirements.txt
cd SmartHomeHARLib
python setup.py develop
```

## Embeddings Training

To train Embedding model(s) of the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Activity Sequences Classification Training And Evaluation

To train Classifier(s) model(s) of the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Results

Our model achieves the following performance on :

### [Three CASAS datasets](http://casas.wsu.edu/datasets/)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

