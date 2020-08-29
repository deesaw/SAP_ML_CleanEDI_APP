# -*- coding: utf-8 -*-
##Import Libraries
def model_train(filename=''):
    print('Getting Trained')
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    import pickle
    from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,classification_report,f1_score
    from sklearn.compose import ColumnTransformer
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import os

    #filepath = os.path.join('dataset','Train.tsv')
    filepath = os.path.join('dataset', filename)
    if os.path.isfile(filepath):
        print('udp', filepath)
        df_raw = pd.read_csv(filepath, sep='\t',dtype=object)
    else:
        print('abc')
        df_raw = pd.read_csv(r'dataset/Train.tsv', sep='\t',dtype=object)
    df = df_raw.copy()
    print(df.columns)
    df.columns=['LIFNR', 'MAT1', 'MAT2', 'MAT3', 'SP_KUNNR', 'MATNR']
    df=df[df.duplicated(subset=["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"],keep='last')]
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    X1,y1=df[["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"]],df["MATNR"]
    #Label Encoding
    label_encoder_y = LabelEncoder()
    y= label_encoder_y.fit_transform(y1)
    pickle.dump(label_encoder_y, open('model_files/01le_cleanEDI.sav', 'wb'))

    #OnehotEncoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [0,1,2,3,4])], remainder='passthrough')
    X = np.array(ct.fit_transform(X1))
    pickle.dump(ct,open('model_files/02ohe_cleanEDI.sav', 'wb'))

    #Training Model
    classifier = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0)
    classifier.fit(X, y)
    pickle.dump(classifier,  open('model_files/03randomforest_cleanEDI.sav', 'wb'))

    #Score
    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)
    y_conf = [(np.max(pred_err)*100) for pred_err in y_prob]
    print(y_conf)
    cm = confusion_matrix(y, y_pred)
    #cm = confusion_matrix(y, classifier.predict(X))
    print( accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
    y_actual=label_encoder_y.inverse_transform(y)
    return(accuracy_score)



if __name__ == "__main__":
    model_train()
