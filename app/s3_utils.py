import os
import boto3

def getS3Client():
    return boto3.client('s3', 
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), 
                      region_name=os.getenv('REGION_NAME')
                      )

def download_data_from_s3(fileKey):
    s3 = getS3Client()
    try:
        s3.download_file(os.getenv('S3_BUCKET_NAME'), fileKey, fileKey)
        print(f"File {fileKey} downloaded successfully.")
        return True
    except Exception as e:
        print(f"Error downloading file {fileKey} from S3: {str(e)}")
        return False

def download_model_from_s3():
    download_data_from_s3('xgboost_model.pkl')
    
def save_to_s3(file, key):
    s3 = getS3Client()
    try:
        s3.put_object(Bucket=os.getenv('S3_BUCKET_NAME'), Key=key, Body=file)
        print(f"{key} saved successfully to S3.")
        return True
    except Exception as e:
        print(f"Error saving {key} to S3: {str(e)}")
        return False

