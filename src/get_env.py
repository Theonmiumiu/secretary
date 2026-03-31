import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_API_URL = os.getenv("LLM_API_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")
    REVIEW_MODEL = os.getenv('REVIEW_MODEL')

    MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
    TEMPERATURE = float(os.getenv("TEMPERATURE"))
    TOP_P = float(os.getenv("TOP_P"))
    TOP_K = int(os.getenv("TOP_K"))
    FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY"))
    MAX_RETRY = int(os.getenv("MAX_RETRY"))

config = Config()