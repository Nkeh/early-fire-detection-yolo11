# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="xGXvprcgEcDhhAphPosq"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="ransomworkspace",
    workflow_id="detect-count-and-visualize-2",
    images={
        "image": "/content/imagesfire.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)