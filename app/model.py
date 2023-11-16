from datetime import datetime
import json
import os
import pickle
import numpy as np
import concurrent.futures
from app.train import train_model
from .s3_utils import download_data_from_s3, download_model_from_s3, getS3Client

latest_train_timestamp = None

def load_model():
    global latest_train_timestamp
    modelPath = 'xgboost_model.pkl'
    if os.path.exists(modelPath):
        if latest_train_timestamp == None:
            modification_time = os.path.getmtime(modelPath)
            last_modified_date = datetime.fromtimestamp(modification_time).strftime("%Y%m%d_%H%M%S")
            latest_train_timestamp = last_modified_date
        with open(modelPath, 'r') as file:
            return file
    else:
        print("Model local file does not exist, download from s3")
        return download_model_from_s3()

def save_model_to_s3(model):
    s3 = getS3Client()
    
    try:
        serialized_model = pickle.dumps(model)
        s3.put_object(Bucket=os.getenv('S3_BUCKET_NAME'), Key='xgboost_model.pkl', Body=serialized_model)
        print("Model saved successfully to S3.")
        return True
    except Exception as e:
        print(f"Error saving model to S3: {str(e)}")
        return False
    
def predict(json_data):
    data_list = json.loads(json_data)
    # Convert list to NumPy array and reshape it to (1, 30)
    numpy_array = np.array(data_list).reshape(1, 30)
    return model_cache.predict(numpy_array)

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

    combined_data.to_csv('./creditcard.csv', index=False)

    model = train_model()
    with open('xgboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)
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