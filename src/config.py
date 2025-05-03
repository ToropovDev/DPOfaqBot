from dotenv import load_dotenv
import os

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

if BOT_TOKEN is None:
    raise ValueError("Не найден BOT_TOKEN в .env файле")

EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
GENERATION_MODEL_NAME = "models/finetuned_rugpt"
