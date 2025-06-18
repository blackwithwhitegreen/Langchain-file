from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# Debug: Print to confirm it's loaded
print("KEY:", os.getenv("GOOGLE_API_KEY"))
