# Sentiment Analysis

Sentiment analysis neural network trained by fine-tuning [BERT](https://arxiv.org/pdf/1810.04805.pdf), [ALBERT](https://arxiv.org/pdf/1909.11942.pdf), or [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) on the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/).

![https://i.imgur.com/XPQFCix.mp4](https://github.com/barissayil/SentimentAnalysis/blob/master/public/analyze.gif)

## Setup Sentiment Analysis

### Clone Sentiment Analysis, update lists of packages, install Pyton 3.9
```
git clone https://github.com/barissayil/SentimentAnalysis.git
cd SentimentAnalysis
sudo apt-get update
sudo apt-get install python3.9
```

### Verify that you have Python 3.9.10
```
python3.9 --version
```

### Create virtual environment with Python 3.9.10 binary, activate it
```
python3.9 -m venv env
source env/bin/activate
```

### Install necessary packages
```
pip install -r requirements.txt
```

### Test it and verify that it passes all tests
```
python -m pytest
```

## Use Sentiment Analysis with my model

### Evaluate
```
python evaluate.py
```

### Analyze your inputs
```
python analyze.py
```

### Run server, and verify that you get back 99% positive
```
python server.py
```
```
curl localhost:5000 -G -d text=good
```

## Train your own model and use Sentiment Analysis with it

### Train (i.e.fine-tune) BERT
```
python train.py --model_name_or_path bert-base-uncased --output_dir XXX --num_eps 2
```
*bert-base-uncased, albert-base-v2, distilbert-base-uncased, and other similar models are supported.*

### Evaluate
```
python evaluate.py --model_name_or_path XXX
```

### Analyze your inputs
```
python analyze.py --model_name_or_path XXX
```

### Run server
```
python server.py --model_name_or_path XXX
```
