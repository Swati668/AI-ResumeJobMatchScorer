import streamlit as st
from sentence_transformers import SentenceTransformer
from flashtext import KeywordProcessor



import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
csv_path = os.path.join(BASE_DIR, 'skills.csv')  
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"skills.csv not found at {csv_path}")

df = pd.read_csv(csv_path, header=None, names=['Skill'])
#df=pd.read_csv('modules/skills.csv',header=None,names=['Skill'])
skills_list = df['Skill'].dropna().str.lower().str.strip().unique().tolist()


    
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_kp(skills_list, skill_mapping):
    kp = KeywordProcessor(case_sensitive=False)
    
    for key, value in skill_mapping.items():
        kp.add_keyword(key, value)
    
    kp.add_keywords_from_list(skills_list)
    
    return kp

