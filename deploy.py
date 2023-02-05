from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.estimator import SKLearnModel

role = 'acgirish'

# Create the SKLearn Object by directing it to the aws_sklearn_main.py script
"""aws_sklearn = SKLearn(source_dir='./',
                      entry_point='aws_training.py',
                      instance_type='ml.m5.4xlarge',
                      role=role,
                      framework_version="1.0-1",
                      py_version='py3')"""


# old: train_instance_type = ml.m4.xlarge
# ml.t2.xlarge
# ml.p2.xlarge
# ml.t2.medium
#
# framework_version="0.23-1"
#framework_version="0.24",

# Train the model using by passing the path to the S3 bucket with the training data

#aws_sklearn.fit({'train': 's3://sagemaker-newsy-api/train/'})

# Deploy model
# old instance type - ml.m4.xlarge
# ml.p2.xlarge

model_data = "s3://sagemaker-us-east-2-482523031755/sagemaker-scikit-learn-2023-01-02-00-16-10-673/output/model.tar.gz"
source_dir = "./"
#source_dir = "s3://sagemaker-us-east-2-482523031755/sagemaker-scikit-learn-2023-01-02-00-16-10-673/source/sourcedir.tar.gz"
aws_sklearn_model = SKLearnModel(model_data= model_data,
                             role=role,
                             entry_point="aws_training.py",
                             source_dir=source_dir,
                             framework_version="1.0-1")

aws_sklearn_predictor = aws_sklearn_model.deploy(instance_type='ml.m5.4xlarge',initial_instance_count=1)
# initial_instance_count=1

# Print the endpoint to test in next step
print(aws_sklearn_predictor.endpoint)

aws_sklearn_predictor.delete_endpoint()
# Uncomment and run to terminate the endpoint after you are finished