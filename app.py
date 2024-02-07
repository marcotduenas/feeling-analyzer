from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np


app = Flask(__name__)
model = load_model('./model_development/rating_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    answers = ["positive", "negative"]
    user_rating = float(request.form['comment'])
    pred = model.predict(np.array([[user_rating]]))
    print(pred)

    if pred > 0.95:
        return render_template('index.html', result=answers[0])
    else:
        return render_template('index.html', result=answers[1])

if __name__ == '__main__':
    app.run(debug=True)
