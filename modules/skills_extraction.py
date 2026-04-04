from modules.model_utils import get_kp,skills_list

import pandas as pd
import re



# relevant_keywords = [
#     'machine learning',
#     'deep learning',
#     'data science',
#     'data analysis',
#     'natural language processing',
#     'computer vision',
#     'artificial intelligence',
#     'python',
#     'sql',
#     'pandas',
#     'numpy'
# ]
# filtered_df=df[df['label_cleaned'].str.contains('|'.join(relevant_keywords),case=False,na=False)]
# skills_list=(filtered_df['label_cleaned'].dropna().str.lower().str.strip().unique().tolist())
# df=pd.Dataframe(skills_list)

#df.to_csv('modules/skills.csv',index=False)

# print(df)

def clean_skill(skill):
    skill=skill.lower().strip()
    remove_words=['perform','produce','use','develop','principles of','utilise','experience in','knowledge of','familiar with']
    pattern=r'\b(' + '|'.join(remove_words) + r')\b'
    skill=re.sub(pattern,' ',skill)
    skill=re.sub(r'\s+',' ',skill)
    return skill.strip()

def normalize_skill(skill):
    mapping = {
        'computer vision system': 'computer vision',
        'develop computer vision system': 'computer vision',
        
        'online data analysis': 'data analysis',
        'methods of logistical data analysis': 'data analysis',
        'specific data analysis software': 'data analysis',
        'safety data analysis': 'data analysis',
        'perform data analysis': 'data analysis',
        
        'principles of artificial intelligence': 'artificial intelligence',
        'utilise machine learning': 'machine learning'
    }
    
    return mapping.get(skill, skill)#if skill not in mapping return skill
 

skills_list = [clean_skill(skill) for skill in skills_list]
skills_list=[normalize_skill(skill) for skill in skills_list]
skills_list=list(set(skills_list))
print(skills_list)



# skill_mapping = {
#     'ml': 'machine learning',
#     'machine learning': 'machine learning',
#     'deep learning': 'deep learning',
#     'ai': 'artificial intelligence',
#     'artificial intelligence': 'artificial intelligence',
#     'nlp': 'natural language processing',
#     'computer vision': 'computer vision',
#     'cv': 'computer vision',
#     'sql': 'sql',
#     'mysql': 'sql',
#     'postgresql': 'sql',
#     'python': 'python'
# }
skill_mapping = {
    'ml': 'machine learning',
    'machine learning': 'machine learning',
    'deep learning': 'deep learning',
    'ai': 'artificial intelligence',
    'artificial intelligence': 'artificial intelligence',
    'nlp': 'natural language processing',
    'natural language processing': 'natural language processing',
    'tokenization': 'natural language processing',
    'embeddings': 'natural language processing',
    'text classification': 'natural language processing',
    'computer vision': 'computer vision',
    'cv': 'computer vision',
    'sql': 'sql',
    'mysql': 'sql',
    'postgresql': 'sql',
    'sql server': 'sql',
    'nosql': 'nosql',
    'python': 'python',
    'pytorch': 'pytorch',
    'torch': 'pytorch',
    'tensorflow': 'tensorflow',
    'tf': 'tensorflow',
    'flask': 'flask',
    'fastapi': 'fastapi',
    'streamlit': 'streamlit',
    'aws': 'cloud',
    'gcp': 'cloud',
    'azure': 'cloud',
    'git': 'version control',
    'version control': 'version control'
}

#extracting_keyword

kp=get_kp(skills_list,skill_mapping)

def extract_skills(text):
    return list(set(kp.extract_keywords(text)))

