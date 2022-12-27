from sagemaker.sklearn.estimator import SKLearn

role = 'acgirish'

# Create the SKLearn Object by directing it to the aws_sklearn_main.py script
aws_sklearn = SKLearn(source_dir='./',
                      entry_point='aws_training.py',
                      instance_type='ml.m5.4xlarge',
                      role=role,
                      framework_version="1.0-1",
                      py_version='py3')


# old: train_instance_type = ml.m4.xlarge
# ml.t2.xlarge
# ml.p2.xlarge
# ml.t2.medium
#
# framework_version="0.23-1"
#framework_version="0.24",

# Train the model using by passing the path to the S3 bucket with the training data
aws_sklearn.fit({'train': 's3://sagemaker-newsy-api/train/'})

# Deploy model
# old instance type - ml.m4.xlarge
# ml.p2.xlarge
aws_sklearn_predictor = aws_sklearn.deploy(instance_type='ml.m5.4xlarge',
                                           initial_instance_count=1,
                                           model_server_workers=2)

# Print the endpoint to test in next step
print(aws_sklearn_predictor.endpoint)
#predictor.delete_endpoint()
# Uncomment and run to terminate the endpoint after you are finished