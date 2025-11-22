from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


my_repo_id = "harishsohani/Bank-Customer-Churn"
my_repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=my_repo_id, repo_type=my_repo_type)
    print(f"Space '{my_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{my_repo_id}' not found. Creating new space...")
    create_repo(repo_id=my_repo_id, repo_type=my_repo_type, private=False)
    print(f"Space '{my_repo_id}' created.")

api.upload_folder(
    folder_path="week2_mlops/data",
    repo_id=my_repo_id,
    repo_type=my_repo_type,
)
