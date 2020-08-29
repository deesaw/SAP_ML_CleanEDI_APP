# -*- coding: utf-8 -*-


# import the necessary packages
import numpy as np
import flask
from flask import current_app, Flask, render_template, request
import io
import pickle
import pandas as pd
from io import StringIO
from pandas.io.json import json_normalize
from random_forest import model_train
import csv
import os

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def load_model():
    global modelr, ohe_risk_catr, le_enc_tgtr
    modelr = pickle.load(open('model_files/03randomforest_cleanEDI.sav', 'rb'))
    ohe_risk_catr = pickle.load(open('model_files/02ohe_cleanEDI.sav', 'rb'))
    le_enc_tgtr = pickle.load(open('model_files/01le_cleanEDI.sav', 'rb'))

def prepare_data(X_test):
    X_test.columns = ['LIFNR', 'MAT1', 'MAT2', 'MAT3', 'SP_KUNNR']
    X_test= pd.DataFrame(ohe_risk_catr.transform(X_test))
    return X_test

def train_model(filename):
    print('deepa', filename)
    model_train(filename)
    return render_template('form1.html')

@app.route("/", methods=["GET"])
def home_page():
    return render_template('form.html')

@app.route("/template", methods=["POST"])
def template():
    if flask.request.method == "POST":
        return render_template('form.html')

@app.route("/train", methods=["POST"])
def train():
    if flask.request.method == "POST":
        if flask.request.files.get("csv_file"):
            file = flask.request.files["csv_file"]
            file.save(os.path.join('dataset',file.filename))
            #df_train = pd.read_csv(file.filename)
            filename = file.filename
            train_model(filename)
            print('Training successful')
            render_template('form1.html')
        return render_template('form.html')



@app.route("/predict", methods=["GET","POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.form:
            cols=[]
            vals = []
            dict_val = flask.request.form.to_dict()
            print(dict_val)
            df_test = pd.DataFrame(dict_val, index=[0])
            print(df_test)
            df_test1 = df_test.copy()
            df_test.columns = ['LIFNR', 'MAT1', 'MAT2', 'MAT3', 'SP_KUNNR']
            df_test = pd.DataFrame(ohe_risk_catr.transform(df_test))
            #df_test = prepare_data(df_test)
            print(df_test.shape)
            predss = modelr.predict(df_test)
            preds=le_enc_tgtr.inverse_transform(predss)
            prob = modelr.predict_proba(df_test)
            conf = (np.max(prob[0])*100)
            print(prob)
            print(preds[0])
            print(conf)
            df_test1['MATNR']= preds[0]
            return render_template('view.html',predictions=df_test1, flag1=True, res = preds[0], conf=conf)
    return render_template('form.html')

@app.route("/predict_json", methods=["GET","POST"])
def predict_json():
    if flask.request.method == "POST":
        print('1')
        if flask.request.get_json():
            json_data = flask.request.get_json()
            df_test = pd.DataFrame.from_dict(json_data, orient="index")
            df_test1 = df_test.copy()
            df_test.columns = ['LIFNR', 'MAT1', 'MAT2', 'MAT3', 'SP_KUNNR']
            df_test = pd.DataFrame(ohe_risk_catr.transform(df_test))
            preds = le_enc_tgtr.inverse_transform(modelr.predict(df_test))
            print(preds)
            df_test1['MATNR'] = preds
            parsed = df_test1.to_json(orient="index")
            return parsed
    return 'Hello, Please pass required data in JSON format for prediction results'
@app.route("/predict_mass", methods=["GET","POST"])
def predict_mass():
    if flask.request.method == "POST":
        if flask.request.files.get("csv_file"):
            file = flask.request.files["csv_file"]
            bytes_data = file.read()
            df_test = pd.read_csv(file.filename)
            df_test1 = df_test.copy()
            print(df_test)
            df_test = prepare_data(df_test)
            preds = le_enc_tgtr.inverse_transform(modelr.predict(df_test))
            print(preds)
            df_test1['ML predictions'] = preds
            return render_template('form.html',predictions=df_test1.values, flag=True)
    return render_template('form.html')
if __name__ == "__main__":
	print(("* Loading models and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0',debug=True,port=5000)

'''


@app.route('/predictsap', methods=["GET","POST","PUT"])
def summary():
    if request.method == "POST":
        if request.get_json():
            req = request.get_json()
            dataframe = pd.DataFrame.from_dict(req, orient="index")
            print(dataframe)
            file = open("models.pkl", "rb")
            trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder
            trained_model_for_prediction = pk.load(file)
            label_encoder_yy= pk.load(file)
            y_predict=trained_encoder.transform(dataframe)
            y_pred=trained_model_for_prediction.predict(y_predict)
            y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
            file.close()
            dataframe['MATNR']=y_pred_oo_pred
            dataframe=dataframe.to_json(orient='index')
            return dataframe, 200
'''
