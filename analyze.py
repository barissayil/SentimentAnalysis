from arguments import args
from analyzer import Analyzer


if __name__ == "__main__":

	print('Please wait while the analyzer is being initialized.')

	analyzer = Analyzer(will_train=False, args=args)
	
	text = input('Input text to analyze sentiment: ')

	while text:
		sentiment, percentage = analyzer.classify_sentiment(text)
		print(f"{sentiment} sentiment with {percentage}% probability.")
		text = input('Input sentiment to analyze: ')