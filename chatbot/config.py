import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# علشان مكتبة langchain_openai ما تلخبطش وتستخدم OpenRouter بدل OpenAI
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"


DATA_DIR = os.path.join(os.path.dirname(__file__), "user_data")
os.makedirs(DATA_DIR, exist_ok=True)
