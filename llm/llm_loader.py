import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def load_gemini_llm(model_name="gemini-2.0-flash"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel(
        model_name=model_name
    )
    return model

