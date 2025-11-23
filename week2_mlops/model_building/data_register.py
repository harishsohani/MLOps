
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo, login
import os

login(token=os.getenv("HF_TOKEN"))

# ------------------------------
# Configuration
# ------------------------------
my_repo_id = "harishsohani/Bank-Customer-Churn"
my_repo_type = "dataset"
my_token = os.getenv("HF_TOKEN")

# ------------------------------
# Validate token
# ------------------------------
if not my_token:
    raise ValueError("HF_TOKEN is not set in environment variables!")


# Initialize API client
#api = HfApi(token=os.getenv("HF_TOKEN"))
# Init client
api = HfApi(token=my_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=my_repo_id, repo_type=my_repo_type)
    print(f"Dataset '{my_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{my_repo_id}' not found. Creating new space...")
    api.create_repo(repo_id=my_repo_id, repo_type=my_repo_type, private=False)
    print(f"Dataset '{my_repo_id}' created.")

# Upload data
api.upload_folder(
    folder_path="week2_mlops/data",
    repo_id=my_repo_id,
    repo_type=my_repo_type,
)
print("Upload completed!")
