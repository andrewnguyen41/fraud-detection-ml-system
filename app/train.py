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

def train_model():
    df = pd.read_csv('creditcard.csv')
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

    start_time = time.time()

    if os.path.exists('xgboost_model.pkl'):
        with open('xgboost_model.pkl', 'r') as file:
            model = pickle.load(file)
    else:
        model = xgb.XGBClassifier(n_estimators=5000, max_depth=30, learning_rate=0.01)
    model.fit(x_train_s, y_train_s)
    y_pred = model.predict(x_test)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")
    print(classification_report(y_test, y_pred))
    return model