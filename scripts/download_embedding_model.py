from huggingface_hub import snapshot_download
snapshot_download('google/flan-t5-xl', allow_patterns=['config.json', '*.safetensors', '*.safetensors.index.json', '*token*', 'spiece.model'], max_workers=4)
