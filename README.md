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

To train a Word2Vec model on a dataset, run this command:
```w2v_train
python Word2vecEmbeddingExperimentations.py --d cairo
```

To train a ELMo model on a dataset, run this command:
```elmo_train
python ELMoEmbeddingExperimentations.py --d milan
```

## Activity Sequences Classification Training And Evaluation

To train Classifier(s) model(s) of the paper, run this command:

```classifier
python PretrainEmbeddingExperimentations.py --d cairo --e bi_lstm --c config/no_embedding_bi_lstm.json
```

```classifier_liciotti_train
python PretrainEmbeddingExperimentations.py --d cairo --e liciotti_bi_lstm --c config/liciotti_bi_lstm.json
```

```classifier_w2v_train
python PretrainEmbeddingExperimentations.py --d cairo --e w2v_bi_lstm --c config/cairo_bi_lstm_w2v.json
```

```classifier_elmo_train
python PretrainEmbeddingExperimentations.py --d cairo --e elmo_bi_lstm --c config/cairo_bi_lstm_elmo.json
```

## Results

Our model achieves the following performance on :

### [Three CASAS datasets](http://casas.wsu.edu/datasets/)

|                    |     Aruba    |    Aruba   | Aruba | Aruba |     Milan    |    Milan   | Milan | Milan |     Cairo    |    Cairo   | Cairo | Cairo |
|                    | No Embedding | Eembedding |  W2V  |  ELMo | No Embedding | Eembedding |  W2V  |  ELMo | No Embedding | Eembedding |  W2V  |  ELMo |
|:------------------:|:------------:|:----------:|:-----:|:-----:|:------------:|:----------:|:-----:|:-----:|:------------:|:----------:|:-----:|:-----:|
|      Accuracy      |     95.01    |    96.56   | 96.59 | 97.02 |     82.24    |    90.68   | 88.33 | 90.89 |     81.68    |    86.17   | 82.27 | 90.66 |
|      Precision     |     94.69    |    96.16   | 96.23 | 96.63 |     82.28    |    90.03   | 88.28 | 90.82 |     80.22    |    85.41   | 82.04 | 90.84 |
|       Recall       |     95.01    |    96.56   | 96.59 | 97.03 |     82.24    |    90.68   | 88.33 | 90.89 |     81.68    |    86.17   | 82.27 | 90.66 |
|      F1 score      |     94.74    |    96.28   | 96.32 | 96.75 |     81.97    |    90.14   | 87.98 | 90.68 |     80.49    |    85.53   | 81.14 | 90.54 |
|  Balance Accuracy  |     77.73    |    78.84   | 81.06 |   81  |     67.77    |    74.73   | 73.61 | 80.15 |     70.09    |    78.18   | 69.38 | 85.21 |
| Weighted Precision |     79.75    |    80.63   | 82.97 |  84.6 |     79.6     |    79.86   | 84.42 | 89.09 |     68.45    |    79.18   | 77.56 | 88.75 |
|   Weighted Recall  |     77.73    |    78.84   | 81.06 | 81.01 |     67.77    |    74.73   | 73.62 | 80.15 |     70.09    |    78.18   | 69.38 | 85.21 |
|  Weighted F1 score |     77.92    |    79.25   | 81.43 | 82.18 |     71.81    |    76.51   | 76.59 | 83.27 |     68.47    |    78.19   | 70.95 | 86.48 |

