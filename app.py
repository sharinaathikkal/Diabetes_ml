import numpy as np
from sklearn.utils.validation import check_array
import pandas as pd
from flask import Flask,request,render_template,redirect
import pickle
app=Flask(__name__)
#diab=pd.read_csv('diabetes(1).csv')
model=pickle.load(open('model.pkl','rb'))
@app.route('/',methods=['GET','POST'])
def home():
    return render_template("index.html")
@app.route('/Predict',methods=['POST'])
def predict():
    bmi=request.form['bmi']
    glucose =request.form['glucose']
    cholesterol=request.form['cholesterol']
    height=request.form['height']
    weight=request.form['weight']
    hdl_chol= request.form['hdl_chol']
    systolic_bp=request.form['systolic_bp']
    diastolic_bp = request.form['diastolic_bp']
    chol_hdl_ratio=request.form['chol_hdl_ratio']
    waist_hip_ratio=request.form['waist_hip_ratio']

    bmi=float(bmi)
    glucose=int(glucose)
    cholesterol=int(cholesterol)
    height=int(height)
    weight=int(weight)
    hdl_chol=int(hdl_chol)
    systolic_bp=int(systolic_bp)
    diastolic_bp=int(diastolic_bp)
    chol_hdl_ratio=float(chol_hdl_ratio)
    waist_hip_ratio=float(waist_hip_ratio)

    feature_names=np.array([(bmi,glucose,cholesterol,height,weight,hdl_chol,systolic_bp,diastolic_bp,chol_hdl_ratio,waist_hip_ratio)])
    array = check_array(feature_names)
    print(array)
    prediction=model.predict(feature_names)
    return render_template("index.html",prediction_text='You have diabetes:{}'.format(prediction))
if __name__=="__main__":
    app.run(debug=True)