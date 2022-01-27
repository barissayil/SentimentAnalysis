from flask import Flask, jsonify, request
from flask_cors import CORS

from arguments import args
from analyzer import Analyzer


app = Flask(__name__)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

analyzer = Analyzer(will_train=False, args=args)

@app.get('/')
def sentiment():
	text = request.args['text']
	sentiment, percentage = analyzer.classify_sentiment(text)
	return jsonify({'sentiment': sentiment, 'percentage': percentage})

if __name__ == '__main__':
	app.run()