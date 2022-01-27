import argparse

from analyzer import Analyzer

args = argparse.Namespace()
args.model_name_or_path = 'barissayil/bert-sentiment-analysis-sst'
args.output_dir = 'my_model'

def test_analyzer():
	analyzer = Analyzer(will_train=False, args=args)
	assert isinstance(analyzer, Analyzer)

	assert analyzer.classify_sentiment('good') == ('Positive', 99)
	assert analyzer.classify_sentiment('bad') == ('Negative', 99)
	assert analyzer.classify_sentiment('average') == ('Negative', 68)