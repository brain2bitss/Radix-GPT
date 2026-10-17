import os


def setup_hf_token():
    """Make HF_TOKEN available from the environment or Kaggle Secrets (secret named HF_TOKEN)."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        return True
    except Exception as e:
        print(f"Could not load HF_TOKEN ({e}). Hub upload/download will be skipped or fail.")
        return False
