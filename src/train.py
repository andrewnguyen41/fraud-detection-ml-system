import os
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report

def preprocess_value(json):
    # preprocessing
    rob_scaler = pickle.load(open('./models/rob_scaler.pkl', 'rb'))
    print(json)
    json['scaled_amount'] = rob_scaler.transform([[json['Amount']]])[0][0]
    json['scaled_time'] = rob_scaler.transform([[json['Time']]])[0][0]
    keys_order = [
    "scaled_amount", "scaled_time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
    ]
    values = [json[key] for key in keys_order]
    numpy_array = np.array(values, dtype=np.float64).reshape(1, -1)
    print(numpy_array)
    return numpy_array

def train_model():
    print("Start training model")
    start_time = time.time()

    df = pd.read_csv('./models/creditcard.csv')

    if os.path.exists('./models/rob_scaler.pkl'):
        rob_scaler = pickle.load(open('./models/rob_scaler.pkl', 'rb'))
    else:
        rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    x = np.array(df.iloc[:, df.columns != 'Class'])
    y = np.array(df.iloc[:, df.columns == 'Class'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    sm = SMOTE(random_state=2)
    x_train_s, y_train_s = sm.fit_resample(x_train, y_train.ravel())

    if os.path.exists('./models/xgboost_model.pkl'):
        model = pickle.load(open('./models/xgboost_model.pkl', 'rb'))
    else:
        model = xgb.XGBClassifier(n_estimators=5000, max_depth=30, learning_rate=0.01)
    model.fit(x_train_s, y_train_s)
    model = xgb.XGBClassifier(n_estimators=5000, max_depth=30, learning_rate=0.01)
    model.fit(x_train_s, y_train_s)
    y_pred = model.predict(x_test)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")
    print(classification_report(y_test, y_pred))

    pickle.dump(model, open('./models/xgboost_model.pkl', 'wb'))
    pickle.dump(rob_scaler, open('./models/rob_scaler.pkl', 'wb'))

    return model