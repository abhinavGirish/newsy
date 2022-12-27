import boto3
import pandas as pd

# Load in the deploy_test data
deploy_test = pd.read_csv("deploy_test.csv").values.tolist()

# Format the deploy_test data features
request_body = ""
for sample in deploy_test:
    request_body += ",".join([str(n) for n in sample[1:-1]]) + "|"
request_body = request_body[:-1]

# create sagemaker client using boto3
client = boto3.client('sagemaker-runtime')

# Specify endpoint and content_type
endpoint_name = "endpoint_from_deployed_model_in_step_6"
content_type = "text/csv"

# Make call to endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=request_body
    )

# Print out expected and returned labels
print(f"Expected {', '.join([n[-1] for n in deploy_test])}")
print("Returned:")
print(response['Body'].read())