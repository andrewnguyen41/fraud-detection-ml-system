from datetime import datetime
import json
import os
import pickle
import numpy as np
import concurrent.futures
from src.train import preprocess_value, train_model
from .s3_utils import download_data_from_s3, download_model_from_s3, getS3Client, save_to_s3

latest_train_timestamp = None

def load_model():
    global latest_train_timestamp
    modelPath = './models/xgboost_model.pkl'
    if os.path.exists(modelPath):
        if latest_train_timestamp == None:
            modification_time = os.path.getmtime(modelPath)
            last_modified_date = datetime.fromtimestamp(modification_time).strftime("%Y%m%d_%H%M%S")
            latest_train_timestamp = last_modified_date

        return pickle.load(open(modelPath, 'rb'))
    else:
        print("Model local file does not exist, download from s3")
        return download_model_from_s3()

def save_model_to_s3(model):
    serialized_model = pickle.dumps(model)
    save_to_s3(serialized_model, 'xgboost_model.pkl')    
    
def predict(json_data):
    parsed_data = json.loads(json_data)
    numpy_array = preprocess_value(parsed_data)
    result = model_cache.predict(numpy_array)
    print(result)
    response = {
        "is_fraud": bool(result)
    }
    print(response)
    return response

def retrain_model():
    #combine all new data files into single file to train
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    train_files = []
    for file in csv_files:
        file_path = os.path.join('.', file)
        modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y%m%d_%H%M%S")
        if modified_time > latest_train_timestamp:
            train_files.append(file_path)

    for f in train_files:
        combined_data = combined_data.append(pd.read_csv(f), ignore_index=True)

    combined_data.to_csv('./models/creditcard.csv', index=False)

    model = train_model()
    save_model_to_s3(model)
    model_cache.reloadModel()

def check_for_new_data():
    global latest_train_timestamp

    bucket_name = os.getenv('S3_BUCKET_NAME')
    s3 = getS3Client()
    response = s3.list_objects_v2(Bucket=bucket_name)

    files_to_download = []
    for obj in response.get('Contents', []):
        file_key = obj['Key']
        if file_key.startswith('creditcard_'):
            file_timestamp_str = file_key.split('_')[1]
            file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M%S")
            if latest_train_timestamp is None or file_timestamp > latest_train_timestamp:
                files_to_download.append(file_key)

    # wait for all new file to download before training
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_data_from_s3, file_key): file_key for file_key in files_to_download}
        
        for future in concurrent.futures.as_completed(futures):
            file_key = futures[future]
            try:
                result = future.result()
                # Process the result if needed
            except Exception as e:
                print(f"Exception occurred for {file_key}: {e}")
   
    retrain_model()

class ModelCache:
    def __init__(self):
        self.model = load_model()

    def reloadModel(self):
        global latest_train_timestamp
        latest_train_timestamp = None
        self.model = load_model()

    def predict(self, input_data):
        if not self.model:
            self.reloadModel()
        return self.model.predict(input_data)

model_cache = ModelCache()
