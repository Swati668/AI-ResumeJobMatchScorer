from google import genai
import streamlit as st

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

def load_gemini_llm():
    return client
