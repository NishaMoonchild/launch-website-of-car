import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    #output = round(prediction[0], 2)

    outputt = prediction-6.5
    output=round(outputt[0],2)



    return render_template('index.html', prediction_text='Cost of your used car is  {} lakhs'.format(output))


if __name__ == "__main__":
    app.run(debug=True)#,host="192.168.43.143"
    