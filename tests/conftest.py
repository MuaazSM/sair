import os

# Set before any test imports settings or model clients.
os.environ["SAIR_ENV"] = "test"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
