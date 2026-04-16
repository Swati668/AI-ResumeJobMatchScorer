from modules.model_utils import load_model
from modules.preprocessing import (get_sentences,clean_text_light)
from modules.skills_extraction import extract_skills


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import re
import torch



def tfidf_similarity(clean_jd,clean_resume):
    vectorizer=TfidfVectorizer()
    vectors=vectorizer.fit_transform([clean_jd,clean_resume])

    return cosine_similarity(vectors[0],vectors[1])[0][0]
    



# document level
model=load_model()
def semantic_similarity_doc(light_clean_jd,light_clean_resume):
    jd_embedding=model.encode(light_clean_jd,convert_to_tensor=True)
    resume_embedding=model.encode(light_clean_resume,convert_to_tensor=True)

    return util.cos_sim(jd_embedding,resume_embedding).item()



# sentence level
def semantic_similarity_sent(light_clean_jd,resume_sentences,k):
    jd_embedding=model.encode(light_clean_jd,convert_to_tensor=True)
    resume_embedding=model.encode(resume_sentences,convert_to_tensor=True)

    scores= util.cos_sim(jd_embedding,resume_embedding)
    scores = scores.squeeze(0)
    k = min(k, len(scores))
    top_k_scores = torch.topk(scores, k).values
    return top_k_scores.mean().item()



def semantic_score(light_clean_jd, light_clean_resume, resume_sentences,k):
    doc_score = semantic_similarity_doc(light_clean_jd, light_clean_resume)
    sent_score = semantic_similarity_sent(light_clean_jd, resume_sentences,k)

    return 0.7*doc_score + 0.3*sent_score


#section based similarity 

section_keywords = {
    "skills": ["skills", "technical skills", "tech stack", "tools"],
    "experience": ["experience", "work experience", "internship", "employment", "work history"],
    "projects": ["projects", "personal projects", "academic projects"],
    "education": ["education", "academic", "qualification"]
}



#Jd parsing
def extract_requirements(jd_text):
    pattern = r"(requirements|qualifications)(.*?)(responsibilities|$)"
    
    match = re.search(pattern, jd_text.lower(), re.DOTALL)
    
    if match:
        return match.group(2).strip()
    
    return ""

def extract_responsibilities(jd_text):
    pattern = r"(responsibilities|job description|role)(.*?)(requirements|$)"
    
    match = re.search(pattern, jd_text.lower(), re.DOTALL)
    
    if match:
        return match.group(2).strip()
    
    return ""

def parse_job_description(jd_text):

    requirements_text = extract_requirements(jd_text)
    responsibilities_text = extract_responsibilities(jd_text)

    jd_sections = {
        "skills": extract_skills(requirements_text),
        "experience": responsibilities_text,
        "projects": "",      
        "education": ""      
    }
    return jd_sections



def get_sentences_advanced(text):
    lines = text.split("\n")

    sentences = []

    for line in lines:
        line = line.strip()

        if not line:
            continue
        parts = re.split(r'[.!?]', line)

        for part in parts:
            part = part.strip()
            if len(part) > 3:
                sentences.append(part)

    return sentences


def build_pattern(start_keywords, all_keywords):
    start = "|".join(start_keywords)
    stop = "|".join(all_keywords)

    pattern = rf"({start})(.*?)(?={stop}|$)"
    return pattern

def extract_sections(text):
    text = text.lower()

    sections = {key: "" for key in section_keywords}

    all_keywords = []
    for kws in section_keywords.values():
        all_keywords.extend(kws)

    for section, start_keywords in section_keywords.items():
        pattern = build_pattern(start_keywords, all_keywords)

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            sections[section] = match.group(2).strip()

    return sections

#if no headings
def fallback_skills(text):
    skill_keywords = [
         "python", "java", "c++", "sql", "machine learning", "deep learning",
        "nlp", "data analysis", "pandas", "numpy", "tensorflow", "pytorch",
         "excel", "power bi", "tableau", "aws", "docker"
    ]

    sentences = get_sentences_advanced(text)     
    skill_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        for skill in skill_keywords:
            if skill in sent_lower:
                skill_sentences.append(sent)
                break
    return " ".join(skill_sentences)


def fallback_experience(text):
    experience_keywords = [
        "worked", "developed", "implemented", "designed",
        "led", "managed", "built", "created", "optimized",
        "intern", "company", "organization"
    ]

    sentences = get_sentences_advanced(text)
    exp_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(word in sent_lower for word in experience_keywords):
            exp_sentences.append(sent)

    return " ".join(exp_sentences)

def fallback_projects(text):
    project_keywords = [
        "project", "developed", "built", "created", "designed",
        "application", "system", "model", "website", "app"
    ]

    sentences = get_sentences_advanced(text)
    proj_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(word in sent_lower for word in project_keywords):
            proj_sentences.append(sent)

    return " ".join(proj_sentences)


def fallback_education(text):
    education_keywords = [
        "btech", "b.e", "mtech", "degree", "university", "college",
        "cgpa", "gpa", "school", "education", "academic"
    ]

    sentences = get_sentences_advanced(text)
    edu_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(word in sent_lower for word in education_keywords):
            edu_sentences.append(sent)

    return " ".join(edu_sentences)



# Section-wise semantic hints 
SECTION_HINTS = {
    "skills": [
       "technical skills",
        "programming languages",
        "tools and technologies",
        "frameworks and libraries"
     ],
    "experience": [
        "work experience",
        "internships",
        "industry experience",
        "roles and responsibilities"
    ],
    "projects": [
        "projects built",
        "applications developed",
        "systems designed",
        "models created"
    ],
    "education": [
        "degree",
        "university",
        "academic background",
        "qualifications"
    ]
}


HINT_EMBEDDINGS = {
    section: model.encode(hints, convert_to_tensor=True)
    for section, hints in SECTION_HINTS.items()
}


def semantic_fallback(text, section):
    
    sentences = get_sentences_advanced(text) 
    if not sentences:
        return ""

    sent_embeddings = model.encode(sentences, convert_to_tensor=True)
    hint_embeddings = HINT_EMBEDDINGS[section]

    cosine_scores = util.cos_sim(sent_embeddings, hint_embeddings)

    selected_sentences = []

    for i, scores in enumerate(cosine_scores):
        max_score = scores.max().item()# extracting best match across all hints

        if max_score >= 0.5:
            selected_sentences.append(sentences[i])

    return " ".join(selected_sentences)


def apply_fallbacks(sections, text):
    for section in sections:
        if not sections["skills"]:
           extracted = fallback_skills(text)

        elif not sections["experience"]:
            extracted = fallback_experience(text)

        elif not sections["projects"]:
         extracted = fallback_projects(text)

        elif not sections["education"]:
            extracted = fallback_education(text)

        if len(extracted.split())<5:
            extracted=semantic_fallback(text,section)
        
        sections[section]=extracted
        
    return sections


#section_wise_score
def section_wise_scores(jd_sections, resume_sections,skills_section_score,k):
    
    weights = {
        "skills": 0.40,
        "experience": 0.25,
        "projects": 0.20,
        "education": 0.15
    }

    section_scores = {}
    final_semantic_score = 0
    total_weight=0

    for section in weights:
        if section=='skills':
            continue
        jd_text = jd_sections.get(section, "").strip()
        res_text = resume_sections.get(section, "").strip()

        if not jd_text:
            section_scores[section] = None
            continue

        if not res_text:
            section_scores[section]=0
            total_weight+=weights[section]
            continue

        jd_text_light = clean_text_light(jd_text)
        res_text_light = clean_text_light(res_text)

        res_sentences = get_sentences(res_text_light)

        score = semantic_score(jd_text_light, res_text_light, res_sentences,k)

        section_scores[section] = score
        total_weight+=weights[section]
        final_semantic_score += score * weights[section]
    section_scores['skills']=skills_section_score
    total_weight+=0.40
    if skills_section_score is not None:
        final_semantic_score+=skills_section_score*0.40
    final_semantic_score=final_semantic_score/total_weight if total_weight else 0

    return final_semantic_score, section_scores