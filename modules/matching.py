from modules.model_utils import load_model
from modules.preprocessing import clean_text_light
from modules.skills_extraction import skill_mapping
model=load_model()


from rapidfuzz import fuzz
from sentence_transformers import util
import re
from nltk.tokenize import sent_tokenize
from collections import Counter




def fuzzy_matching(jd_skills,resume_skills,threshold):
    matched_skills=set()
    missing_skills=set()

    for jd_kw in jd_skills:
        found=False
        for res_kw in resume_skills:
            score=fuzz.ratio(jd_kw,res_kw)
            if score>threshold:
                matched_skills.add(jd_kw)
                found=True
                break
        if not found:
            missing_skills.add(jd_kw)
    return matched_skills,missing_skills




def boost_score(jd_kw,resume_tokens,base_score):

    skill_map={
        'machine learning':['deep learning','regression','supervised learning'],
        'artificial intelligence':['neural network','machine learning']
    }

    if jd_kw in skill_map:

        for related in skill_map[jd_kw]:
             skill_tokens=related.split()
             if all(token in resume_tokens for token in skill_tokens):
                base_score+=0.15
                base_score=min(base_score,1.0)
                break

    return base_score

def semantic_matching(jd_skills,resume_skills,resume_tokens,strong_threshold,weak_threshold):

    strong_matched_skills=set()
    weak_matched_skills=set()
    missing_skills_semantic=set()
    
    jd_emb=model.encode(jd_skills,convert_to_tensor=True)
    res_emb=model.encode(resume_skills,convert_to_tensor=True)

    for i,jd_kw in enumerate(jd_skills):

        scores=util.cos_sim(jd_emb[i],res_emb)
        base_score=scores.max().item()

        boosted_score=boost_score(jd_kw,resume_tokens,base_score)

        if base_score>=strong_threshold:
            strong_matched_skills.add((jd_kw,'semantic'))

        elif boosted_score>=strong_threshold:
            strong_matched_skills.add((jd_kw,'semantic'+'-'+'domain knowledge'))

        elif base_score>=weak_threshold:
            weak_matched_skills.add((jd_kw,'weakly matched'))

        else:
            missing_skills_semantic.add(jd_kw)
    return strong_matched_skills,weak_matched_skills,missing_skills_semantic

def semantic_sentence_matching(jd_sentences,resume_sentences,strong_threshold,weak_threshold):
    strong_matched_sent=set()
    weak_matched_sent=set()
    missing_sent=set()

    jd_emb=model.encode(jd_sentences,convert_to_tensor=True)
    res_emb=model.encode(resume_sentences,convert_to_tensor=True)

    for i,jd_sent in enumerate(jd_sentences):
        scores=util.cos_sim(jd_emb[i],res_emb)
        max_score=scores.max().item()
        if max_score>=strong_threshold:
            strong_matched_sent.add(jd_sent)
        elif max_score>=weak_threshold:
            weak_matched_sent.add(jd_sent)

        else:
            missing_sent.add(jd_sent)


    return strong_matched_sent,weak_matched_sent,missing_sent

    
#matching explanation
def generate_explanation(strong_matched_skills,weak_matched_skills,missing_skills):

    explanations=[]
    strong_dict=dict(strong_matched_skills)
    weak_dict=dict(weak_matched_skills)
    all_keywords=set(list(strong_dict.keys())+list(weak_dict.keys())+list(missing_skills))

    for kw in all_keywords:
        if kw in strong_dict:
            match_type = strong_dict[kw]

            if match_type == 'semantic':
                explanations.append({
                    "skill": kw,
                    "match_type": "STRONG",
                    "reason": "Direct semantic similarity found"
                })

            elif match_type == 'semantic-domain':
                explanations.append({
                    "skill": kw,
                    "match_type": "STRONG",
                    "reason": "Matched via domain knowledge"
                })

        elif kw in weak_dict:
            explanations.append({
                "skill": kw,
                "match_type": "WEAK",
                "reason": "Partial semantic similarity"
            })

        else:
            explanations.append({
                "skill": kw,
                "match_type": "MISSING",
                "reason": "No match found"
            })

    return explanations





CRITICAL_KEYWORDS = ["must have", "required", "mandatory", "essential"]
OPTIONAL_KEYWORDS = ["good to have", "preferred", "nice to have", "plus"]

SECTION_KEYWORDS = {
    "critical": ["requirements", "must have", "required qualifications"],
    "optional": ["preferred", "good to have", "nice to have"]
}

WINDOW_SIZE = 5



def clean_text_light(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def contains_skill(skill, text):
    pattern = r'\b' + re.escape(skill) + r'\b'
    return bool(re.search(pattern, text))




def is_near_keyword(text, skill, keywords, window=WINDOW_SIZE):
    words = text.split()

    for i, word in enumerate(words):
        if re.fullmatch(skill, word):  # strict match
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            context = " ".join(words[start:end])

            if any(kw in context for kw in keywords):
                return True
    return False



def detect_sections(jd_text):
    lines = jd_text.lower().split("\n")
    lines = [clean_text_light(l) for l in lines]

    sections = []
    current_section = "general"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Treat SHORT lines as headers
        if len(line.split()) <= 5:
            if any(k in line for k in SECTION_KEYWORDS["critical"]):
                current_section = "critical"
                continue
            elif any(k in line for k in SECTION_KEYWORDS["optional"]):
                current_section = "optional"
                continue

        sections.append((line, current_section))

    return sections




def get_skill_frequency(jd_text, jd_skills):
    skill_freq = {}
    clean_jd_text = clean_text_light(jd_text)

    for skill in jd_skills:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        matches = re.findall(pattern, clean_jd_text)
        skill_freq[skill] = len(matches)

    return skill_freq




def assign_skill_importance(jd_text, missing_skills):

    sentences = sent_tokenize(jd_text)
    sentences = [clean_text_light(s) for s in sentences]

    sections = detect_sections(jd_text)
    skill_freq = get_skill_frequency(jd_text,missing_skills)

    skill_importance = {}

    for skill in missing_skills:
        skill_clean = clean_text_light(skill)

        assigned = False

        
        for line, section in sections:
            if contains_skill(skill_clean, line):
                if section == "critical":
                    skill_importance[skill] = 'critical'
                    assigned = True
                    break
                elif section == "optional":
                    skill_importance[skill] = 'optional'
                    assigned = True
                    break

        
        if not assigned:
            for sent in sentences:
                if contains_skill(skill_clean, sent):

                    if any(kw in sent for kw in CRITICAL_KEYWORDS):
                        skill_importance[skill] = 'critical'
                        assigned = True
                        break

                    elif any(kw in sent for kw in OPTIONAL_KEYWORDS):
                        skill_importance[skill] = 'optional'
                        assigned = True
                        break

       
        if not assigned:
            if is_near_keyword(jd_text.lower(), skill_clean, CRITICAL_KEYWORDS):
                skill_importance[skill] = 'critical'
                assigned = True

            elif is_near_keyword(jd_text.lower(), skill_clean, OPTIONAL_KEYWORDS):
                skill_importance[skill] = 'optional'
                assigned = True

        
        if not assigned:
            freq = skill_freq.get(skill, 0)

            if freq >= 4:
                skill_importance[skill] = 'moderate'
            
            else:
                skill_importance[skill] = 'optional'

    return skill_importance


def split_text(jd_text):
    sentences = sent_tokenize(jd_text)
    
    final_sentences = []
    
    for sentence in sentences:
        parts = re.split(r'\n+', sentence)
        final_sentences.extend(parts)
    
    return [s.strip() for s in final_sentences if s.strip()]


def filter_sentences(sentences):
    ignore_words = ["responsibilities", "required skills", "preferred","must have", "required", "mandatory", "essential","good to have",  "nice to have", "plus"]
    
    return [s for s in sentences if not any(word in s.lower() for word in ignore_words)]


def sentence_score(sentence, skill):
    score = 0
    sentence = sentence.lower()

    if skill in sentence:
        score += 2

    action_words = ["develop", "build", "deploy", "train", "experience", "use"]
    if any(word in sentence for word in action_words):
        score += 2

    generic_words = ["we are looking", "responsibilities", "role"]
    if any(word in sentence for word in generic_words):
        score -= 2

    return score


skill_aliases = {
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "artificial intelligence": ["ai"],
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "sql": ["mysql", "postgresql", "sql server"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud"],
    "azure": ["microsoft azure"],
    "git": ["version control"]
}


def contains_skill(skill, sentence):
    skill = skill.lower()
    sentence = sentence.lower()

    
    pattern = r'\b' + re.escape(skill) + r'\b'
    if re.search(pattern, sentence):
        return True

    #matching aliases like 'ml'='machine learning
    if skill in skill_aliases:
        for alias in skill_aliases[skill]:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, sentence):
                return True

    return False


def best_context(skill,jd_text):
    best_score=-1
    best_sentence=""
    sentences=split_text(jd_text)
    sentences=filter_sentences(sentences)

    for sentence in sentences:
        sentence = sentence.lstrip("- ").strip()
        if contains_skill(skill, sentence):
            score = sentence_score(sentence, skill)
            if score > best_score:
                best_score = score
                best_sentence = sentence
    return best_sentence if best_sentence else "Mentioned in job description"

