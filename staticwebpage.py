from distutils.log import debug
# from unittest import result
from flask import Flask,redirect, url_for, request,render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")

def hello():
    return render_template("test.html")

@app.route('/success', methods=["GET", "POST"])

def success():
    if request.method == 'POST':
        identify = request.form.get('identify')
        age = request.form.get('age')
        work = request.form.get('work')
        preg = request.form.get('preg')
        disease = request.form.getlist('disease')
        disease = "".join(disease)
        other_dis = request.form.get('other_dis')
        other_dis_yes = request.form.get('other_dis_yes')
        meds = request.form.get('meds')
        meds_yes = request.form.get('meds_yes')
        bleed = request.form.get('bleed')
        sleep = request.form.get('sleep')
        food = request.form.get('food')
        exercise = request.form.get('exercise')
        symptoms = request.form.getlist('symptoms')
        symptoms = "".join(symptoms)
        sex = request.form.get('sex')
        input = [identify, work, preg, disease, other_dis, other_dis_yes,meds, meds_yes,bleed, sleep, food, exercise, symptoms, sex]

        # load model
        knn = pickle.load(open('knn_model.pkl','rb'))
        encoder = pickle.load(open('one_hot_encoder.pkl','rb'))

        # preprocessing
        processed_arr = encoder.transform((np.array(input).reshape(-1,14)))

        # Predict
        prediction = knn.predict(processed_arr)
        return render_template("test2.html", prediction=prediction)
        


if __name__ == "__main__":
    app.run(debug=True)