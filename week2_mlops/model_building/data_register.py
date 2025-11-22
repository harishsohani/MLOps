
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# ------------------------------
# Configuration
# ------------------------------
my_repo_id = "harishsohani/Bank-Customer-Churn"
my_repo_type = "dataset"
my_token = os.getenv("HF_TOKEN")

# ------------------------------
# Validate token
# ------------------------------
if not token:
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
    create_repo(repo_id=my_repo_id, repo_type=my_repo_type, exist_ok=True, private=False)
    print(f"Dataset '{my_repo_id}' created.")

# ------------------------------
# Step 2: Upload data folder
# ------------------------------
folder_to_upload = os.path.join("week2_mlops", "data")

if not os.path.exists(folder_to_upload):
    raise FileNotFoundError(f"Data folder not found: {folder_to_upload}")

api.upload_folder(
    repo_id=my_repo_id,
    repo_type=my_repo_type,
    folder_path=folder_to_upload,
)
print("Upload completed!")
