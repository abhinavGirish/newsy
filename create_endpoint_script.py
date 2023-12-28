# Specify your AWS Region
aws_region='us-east-2'

# Role to give SageMaker permission to access AWS services.
sagemaker_role="arn:aws:iam::482523031755:role/acgirish"
#sagemaker_role= "arn:aws:iam::us-east-2:acgirish:role/SageMaker-Full-Access"

from sagemaker import image_uris

# Name of the framework or algorithm
framework='sklearn'
#framework='xgboost' # Example

# Version of the framework or algorithm
#version = 'latest'
version = '1.2-1'
#version = '0.20.0'
#version = '0.90-1' # Example

# Specify an AWS container image. 
container = image_uris.retrieve(region=aws_region, 
                                framework=framework, 
                                version=version)

# Create a variable w/ the model S3 URI
# First, provide the name of your S3 bucket
s3_bucket = 'sagemaker-us-east-2-482523031755' 

# Specify what directory within your S3 bucket your model is stored in
bucket_prefix = 'sagemaker-scikit-learn-2023-01-02-00-16-10-673/output'

# Replace with the name of your model artifact
model_filename = 'model.tar.gz'


# Relative S3 path
model_s3_key = f'{bucket_prefix}/'+model_filename

# Combine bucket name, model file name, and relate S3 path to create S3 model URI
model_url = f's3://{s3_bucket}/{model_s3_key}'                            
                        
from sagemaker.model import Model

model = Model(image_uri=container, 
              model_data=model_url,
              source_dir="/Users/abhinavgirish/Documents/2021-2022/Newsy/",
              env={
                  "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv", 
                  "SAGEMAKER_USE_NGINX": "True", 
                  "SAGEMAKER_WORKER_CLASS_TYPE": "gevent", 
                  "SAGEMAKER_KEEP_ALIVE_SEC": "60", 
                  "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                  "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
                  "SAGEMAKER_REGION": "us-east-2",
                  "SAGEMAKER_PROGRAM": "/opt/ml/model/aws_training.py",
                  "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/"
              },
              role=sagemaker_role)

from datetime import datetime

endpoint_name = f"DEMO-{datetime.utcnow():%Y-%m-%d-%H%M}"
print("EndpointName =", endpoint_name)

initial_instance_count=1
# initial_instance_count=1 # Example

instance_type='ml.m4.xlarge'
# instance_type='ml.m4.xlarge' # Example


model.deploy(
    initial_instance_count=initial_instance_count,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    wait=False
)

