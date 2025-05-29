from dotenv import load_dotenv
import os
# Load environment variables from .env file

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
if key is None:
    print("API_KEY is not set in the environment variables.")
else:
    print(key, "API_KEY is set in the environment variables.")