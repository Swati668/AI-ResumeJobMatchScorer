#import os
#from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st


#load_dotenv()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def load_gemini_llm(model_name="gemini-2.0-flash"):
    model = genai.GenerativeModel(
        model_name=model_name
    )
    return model

