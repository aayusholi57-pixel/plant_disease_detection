import os
import boto3
from botocore.exceptions import NoCredentialsError

def upload_model():
    s3_client = boto3.client('s3')
    
    bucket_name = "plant-disease-dataset-aayush"  # Your S3 bucket name
    local_model_path = "./models/plant_disease_resnet18.pth"
    s3_model_key = "models/plant_disease_resnet18.pth"  # Path inside S3
    
    if not os.path.exists(local_model_path):
        print(f"Error: Model file not found at {local_model_path}")
        return

    print(f"Uploading fine-tuned model to s3://{bucket_name}/{s3_model_key}...")
    
    try:
        s3_client.upload_file(local_model_path, bucket_name, s3_model_key)
        print("Model successfully uploaded to S3!")
    except NoCredentialsError:
        print("Error: AWS credentials not found. Run 'aws configure' first.")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    upload_model()