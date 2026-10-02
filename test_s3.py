import boto3

# Initialize the S3 resource
s3 = boto3.resource('s3')

print("Successfully connected to AWS S3!")
print("Listing your S3 buckets:")

try:
    for bucket in s3.buckets.all():
        print(f" - {bucket.name}")
except Exception as e:
    print(f"Error connecting to S3: {e}")