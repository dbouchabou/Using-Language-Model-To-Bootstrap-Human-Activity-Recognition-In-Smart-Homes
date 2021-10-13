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
python ELMoEmbeddingExperimentations.py --d cairo
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
python PretrainEmbeddingExperimentations.py --d cairo --e elmo_bi_lstm --c config/cairo_bi_lstm_elmo_concat.json
```

## Results

Our model achieves the following performance on :

### [Three CASAS datasets](http://casas.wsu.edu/datasets/)

|                    |     Aruba    |    Aruba   | Aruba | Aruba |     Milan    |    Milan   | Milan | Milan |     Cairo    |    Cairo   | Cairo | Cairo |
|:------------------:|:------------:|:----------:|:-----:|:-----:|:------------:|:----------:|:-----:|:-----:|:------------:|:----------:|:-----:|:-----:|
|                    | No Embedding |  Liciotti  |  W2V  |  ELMo | No Embedding |  Liciotti  |  W2V  |  ELMo | No Embedding |  Liciotti  |  W2V  |  ELMo |
|      Accuracy      |     95.01    |    96.52   | 96.59 | 96.76 |     82.24    |    90.54   | 88.33 | 90.14 |     81.68    |    84.99   | 82.27 | 90.12 |
|      Precision     |     94.69    |    96.11   | 96.23 | 96.43 |     82.28    |    90.08   | 88.28 | 90.20 |     80.22    |    83.17   | 82.04 | 88.41 |
|       Recall       |     95.01    |    96.50   | 96.59 | 96.69 |     82.24    |    90.45   | 88.33 | 90.31 |     81.68    |    82.98   | 82.27 | 87.59 |
|      F1 score      |     94.74    |    96.22   | 96.32 | 96.42 |     81.97    |    90.02   | 87.98 | 90.10 |     80.49    |    82.18   | 81.14 | 87.48 |
|  Balance Accuracy  |     77.73    |    79.96   | 81.06 | 79.98 |     67.77    |    74.31   | 73.61 | 78.25 |     70.09    |    77.52   | 69.38 | 87.00 |
| Weighted Precision |     79.75    |    82.30   | 82.97 | 88.64 |     79.6     |    82.03   | 84.42 | 87.56 |     68.45    |    80.03   | 77.56 | 86.83 |
|   Weighted Recall  |     77.73    |    80.71   | 81.06 | 79.17 |     67.77    |    75.51   | 73.62 | 78.75 |     70.09    |    73.82   | 69.38 | 84.78 |
|  Weighted F1 score |     77.92    |    81.21   | 81.43 | 82.93 |     71.81    |    77.74   | 76.59 | 82.26 |     68.47    |    74.84   | 70.95 | 84.71 |

