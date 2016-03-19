# import the Flask class from the flask module
from flask import Flask, render_template, make_response
import pandas as pd
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# create the application object
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')  # render a template

@app.route('/corr')
def correlation():
    return render_template('correlation.html')

@app.route('/user_review')
def user_review():
    return render_template('user_review.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/example')
def example():
    return render_template('example.html')

filepath = os.path.join(os.path.dirname(__file__),'static/data/result_rank.csv')

@app.route('/plot')
def plot():
    result_rank = pd.read_csv(filepath)

    return result_rank.to_html()

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
