import joblib
import pandas as pd
import regex as re
import string
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=["POST"])
def predict():
    
    dt_model = open('model_dt.pkl', 'rb')
    clf = joblib.load(dt_model)
    
    cv_model = open('vector.pkl', 'rb')
    cv = joblib.load(cv_model)
    
    wordopt = open('wordOpt2.pkl', 'rb')
    wOPt = joblib.load(wordopt)

    if request.method == "POST":

        message = request.form['berita']
        data = {"text":[message]}
        new_data = pd.DataFrame(data)
        new_data["text"] = new_data["text"].apply(wOPt)
        new_data_test = new_data
        new_vect = cv.transform(new_data_test)
        my_prediction = clf.predict(new_vect)

        if my_prediction == 0:
            result = "Hoax!"
            
        elif my_prediction == 1 :
            result = "Valid."

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
