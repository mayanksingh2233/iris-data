from flask import Flask,render_template,request
import pickle
import numpy as np


model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def home():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    return render_template('after.html',data=prediction)


if __name__=='__main__':
    app.run(debug=True)