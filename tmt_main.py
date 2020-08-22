from flask import *
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle as pk
from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,classification_report,f1_score
#import urllib.parse
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import time
#from flask_restful import Api,Resource
from flask import jsonify
import json

UPLOAD_FOLDER = './uploads_f/'#'/uploads_f'
ALLOWED_EXTENSIONS = set(['tsv','csv'])
pd.set_option('display.max_colwidth', -1)
app = Flask(__name__)
#api=Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        return "Thanks! I did not receive json", 200
    #dataset1 = pd.read_csv('./uploads_f/Test.tsv', sep='\t',dtype=object)
    #dataset1 = dataset1.to_json(orient='index')
    #return dataset1
    return "Thanks for sending a request but that is not a post request"


@app.route('/home', methods=['GET', 'POST'])

def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save('./uploads_f/'+filename)
            f_list = [f for f in listdir('./uploads_f/') if isfile(join('./uploads_f/', f))]
            return render_template('upload1.html',file_list = f_list)
            #file.save('./uploads_f/'+filename)
            #return 'File Uploaded'
            #return redirect(url_for('preview'))	<img src='/static/image.jpeg' style = 'margin:auto; width:70%'></img>
            #                        ,filename=filename))
    #f_list = [f for f in listdir('./') if isfile(join('./', f))]
    f_list = [f for f in listdir('./uploads_f/') if isfile(join('./uploads_f/', f))]
    return render_template('upload.html',file_list = f_list)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

@app.route("/preview")
def preview(filename):
    return 'File Uploaded'

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/predictvalue", methods=['GET', 'POST'])
def variable():
    if request.method == "POST":
        req = request.form
        print(req)
        values=[]
        columnss=[]
        for key, value in req.items():
            print(key, '->', value)
            columnss.append(key)
            values.append(value)
            file = open("models.pkl", "rb")
        print(columnss)
        print(values)
        df = pd.DataFrame([values],columns=columnss)
        print(df)
        trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder
        trained_model_for_prediction = pk.load(file)
        label_encoder_yy= pk.load(file)
        y_predict=trained_encoder.transform(df)
        y_pred=trained_model_for_prediction.predict(y_predict)
        y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
        df['MATNR']=y_pred_oo_pred
        print(df)
        file.close()
        dataset1=df
        df.to_csv('./uploads_f/Predict1.csv')
        return render_template('upload0.html',tables=[dataset1.to_html(classes='dataset1')])
        #return 'Check Console'
    else:
        return 'Check with Expert'


@app.route('/function')
def get_ses():
    #Importing the dataset
    dataseta = pd.read_csv('./uploads_f/Train.tsv', sep='\t',dtype=object)
    dataset=dataseta[dataseta.duplicated(subset=["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"],keep='last')]
    dataset= dataset.head(30000)
    print(dataset.isna().any(axis=0))
    print(dataset.isnull().sum())
    #dataset.dropna()
    print(dataset.info())
    print(dataset.describe())
    X1,y1=dataset[["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"]],dataset["MATNR"]
    label_encoder_y = preprocessing.LabelEncoder()
    y= label_encoder_y.fit_transform(y1)
    file = open("models.pkl", "wb")
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [0,1,2,3,4])], remainder='passthrough')
    X = np.array(ct.fit_transform(X1))
    pk.dump(ct, file) #dumping Encoder model

    # Training the Random Forest Regression model on the whole dataset
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0)
    classifier.fit(X, y)
    pk.dump(classifier, file)
    pk.dump(label_encoder_y,file)
    file.close()
    cm = confusion_matrix(y, classifier.predict(X))
    print( accuracy_score(y, classifier.predict(X)))
    print(classification_report(y, classifier.predict(X)))

    dataset1 = pd.read_csv('./uploads_f/Test.tsv', sep='\t',dtype=object)
    #dataset_test=ct.transform(dataset1)
    file = open("models.pkl", "rb")
    trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder
    trained_model_for_prediction = pk.load(file)
    label_encoder_yy= pk.load(file)
    y_predict=trained_encoder.transform(dataset1)
    y_pred=trained_model_for_prediction.predict(y_predict)
    y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
    dataset1['MATNR']=y_pred_oo_pred
    print(dataset1)
    file.close()
    dataset1.to_csv('./uploads_f/Predict.csv')
    return render_template('upload0.html',tables=[dataset1.to_html(classes='dataset1')])
    #return redirect(url_for('preview'))

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0',debug=True,port=5000)
