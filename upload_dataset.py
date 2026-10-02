import os
import boto3
from botocore.exceptions import NoCredentialsError

def upload_folder_to_s3(local_folder, bucket_name, s3_prefix="data"):
    """
    Recursively uploads a local folder to an S3 bucket.
    """
    s3_client = boto3.client('s3')
    
    if not os.path.exists(local_folder):
        print(f"Error: Local folder '{local_folder}' does not exist!")
        return

    print(f"Starting upload of '{local_folder}' to s3://{bucket_name}/{s3_prefix}/...")
    
    uploaded_files = 0
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            
            # Calculate the relative path to maintain folder structure in S3
            relative_path = os.path.relpath(local_path, local_folder)
            
            # Construct the target S3 key (use forward slashes even on Windows)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            try:
                print(f"Uploading: {relative_path} -> s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_path, bucket_name, s3_key)
                uploaded_files += 1
            except NoCredentialsError:
                print("Error: AWS credentials not found. Please run 'aws configure' first.")
                return
            except Exception as e:
                print(f"Failed to upload {file}: {e}")

    print(f"\nSuccessfully uploaded {uploaded_files} files to S3!")

if __name__ == "__main__":
    # Configure your bucket and local data path here
    BUCKET_NAME = "plant-disease-dataset-aayush"  # Replace with your actual S3 bucket name
    LOCAL_DATA_DIR = "./data"                     # Path to your local dataset folder
    
    upload_folder_to_s3(LOCAL_DATA_DIR, BUCKET_NAME, s3_prefix="data")