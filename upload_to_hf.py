from huggingface_hub import HfApi
import os

# Initialize API
api = HfApi()

# Create repository
try:
    api.create_repo(
        repo_id="kartikshirode/car-price-model",
        private=True,
        exist_ok=True,
        repo_type="model"
    )
    print("✅ Repository created/exists: kartikshirode/car-price-model")
except Exception as e:
    print(f"Repo creation: {e}")

# Upload the model file
print("\n📤 Uploading model file (3GB - this will take a few minutes)...")
try:
    api.upload_file(
        path_or_fileobj="car_model.joblib",
        path_in_repo="car_model.joblib",
        repo_id="kartikshirode/car-price-model",
        repo_type="model"
    )
    print("\n✅ Model uploaded successfully!")
    print("🔗 View at: https://huggingface.co/kartikshirode/car-price-model")
except Exception as e:
    print(f"❌ Upload failed: {e}")
