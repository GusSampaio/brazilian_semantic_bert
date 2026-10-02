from huggingface_hub import HfApi
api = HfApi()

REPO_ID = 'GusSampaio/brazilian-semantic-xlm-roberta-large'

print(f"Starting model upload at: {REPO_ID}")
api.upload_folder(
    folder_path='artifacts/xlm-roberta-large/baseline/seed120/final_model',
    repo_id=REPO_ID,
    repo_type='model'
)
print('Upload finished!')