# Sentiment Analysis

Sentiment analysis neural network trained by fine tuning [BERT](https://arxiv.org/pdf/1810.04805.pdf), [ALBERT](https://arxiv.org/pdf/1909.11942.pdf), or [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/).

## Install requirements
```
pip install numpy pandas torch transformers

```

## Analyze sentiments of your own inputs
```
python analyze.py
```

## Train network
```
python train.py
```

## Evaluate network
```
python evaluate.py
```

## Install requirements for server
```
pip install flask flask_cors
```

## Run server
```
python server.py
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
