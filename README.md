# Sentiment Analysis

Sentiment analysis neural network trained by fine tuning [BERT](https://arxiv.org/pdf/1810.04805.pdf) on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/).

## Install requirements
```
pip3 install -r requirements.txt
```

## Train network
```
python3 train.py
```

## Evaluate network
```
python3 evaluate.py
```

## Analyze sentiments of your own inputs
```
python3 analyze.py
```

## Run server
```
python3 server.py
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
