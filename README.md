# Sentiment Analysis

Sentiment analysis neural network trained by fine-tuning [BERT](https://arxiv.org/pdf/1810.04805.pdf), [ALBERT](https://arxiv.org/pdf/1909.11942.pdf), or [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/).

![https://i.imgur.com/XPQFCix.mp4](https://github.com/barissayil/SentimentAnalysis/blob/master/public/analyze.gif)

## Install requirements
```
pip install numpy pandas torch transformers
```

## Analyze your inputs with the model that I've uploaded on s3
```
python analyze.py
```

## Train model
```
python train.py --model_name_or_path bert-base-uncased --output_dir my_model --num_eps 2
```
*bert-base-uncased, albert-base-v2, distilbert-base-uncased, and other similar models are supported.*

## Evaluate the model that you have trained
```
python evaluate.py --model_name_or_path my_model
```

## Analyze your inputs with the model you have trained
```
python analyze.py --model_name_or_path my_model
```

## Setup server
```
pip install flask flask_cors
```

## Run server
```
python server.py --model_name_or_path my_model
```

## Setup client
```
cd client
npm install
```

## Run client
```
cd client
npm run serve
```
