from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearnModel

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

role = 'acgirish'

#source_dir = "s3://sagemaker-us-east-2-482523031755/sagemaker-scikit-learn-2023-01-02-00-16-10-673/source/sourcedir.tar.gz"
source_dir = "./"

aws_sklearn_model = SKLearnModel(model_data="s3://sagemaker-us-east-2-482523031755/sagemaker-scikit-learn-2023-01-02-00-16-10-673/output/model.tar.gz",
                             role=role,
                             entry_point="aws_training.py",
                             source_dir=source_dir,
                             framework_version="1.0-1")

aws_sklearn_predictor = aws_sklearn_model.deploy(instance_type='local',
                                           initial_instance_count=1)

print(aws_sklearn_predictor.endpoint)

# Tears down the endpoint container and deletes the corresponding endpoint configuration
aws_sklearn_predictor.delete_endpoint()

# Deletes the model
aws_sklearn_predictor.delete_model()