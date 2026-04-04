from modules.model_utils import load_model


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import pandas as pd

df=pd.read_csv('skills.csv')
print(df.columns)



def tfidf_similarity(clean_jd,clean_resume):
    vectorizer=TfidfVectorizer()
    vectors=vectorizer.fit_transform([clean_jd,clean_resume])

    return cosine_similarity(vectors[0],vectors[1])[0][0]
    



# document level
model=load_model()
def semantic_similarity_doc(clean_jd,clean_resume):
    jd_embedding=model.encode(clean_jd,convert_to_tensor=True)
    resume_embedding=model.encode(clean_resume,convert_to_tensor=True)

    return util.cos_sim(jd_embedding,resume_embedding).item()

# sentence level

def semantic_similarity_sent(clean_jd,resume_sentences):
    jd_embedding=model.encode(clean_jd,convert_to_tensor=True)
    resume_embedding=model.encode(resume_sentences,convert_to_tensor=True)

    scores= util.cos_sim(jd_embedding,resume_embedding)
    return scores.max().item()

def semantic_score(clean_jd, clean_resume, resume_sentences):
    doc_score = semantic_similarity_doc(clean_jd, clean_resume)
    sent_score = semantic_similarity_sent(clean_jd, resume_sentences)

    return max(doc_score, sent_score)

#FINDING KEYWORDS FROM JD AND RESUME

def get_keywords(vector,feature_names,top_n):
    indices=vector.argsort()[-top_n:][::-1]
    keywords=[feature_names[i] for i in indices]
    return keywords

def clean_keywords(keywords):
    weak_unigrams=['learning','looking','experience','deep','artificial','neural','knows','network','machine']
    weak_words=['learning','network','knows']
    cleaned=set()
    for kw in keywords:
        words=kw.split()

        if len(words)==1 and words[0] in weak_unigrams:
            continue

        if 'looking'in words or 'experience' in words:
            continue

        if words[0] in weak_words:
            continue

        cleaned.add(kw)
    return cleaned
    



def finding_keywords(clean_jd,clean_resume):

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors=vectorizer.fit_transform([clean_jd,clean_resume])
    feature_names=vectorizer.get_feature_names_out()
    jd_vector=vectors[0].toarray()[0]
    resume_vector=vectors[1].toarray()[0]
    jd_keywords=get_keywords(jd_vector,feature_names,20)
    resume_keywords=get_keywords(resume_vector,feature_names,20)
    return clean_keywords(jd_keywords),clean_keywords(resume_keywords)
    
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
