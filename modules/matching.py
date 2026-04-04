from modules.model_utils import load_model
model=load_model()


from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sentence_transformers import util



def fuzzy_matching(jd_keywords,resume_keywords,threshold):
    matched_skills=set()
    missing_skills=set()

    for jd_kw in jd_keywords:
        found=False
        for res_kw in resume_keywords:
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

def semantic_matching(jd_keywords,resume_sentences,resume_tokens,strong_threshold,weak_threshold):

    strong_matched_skills=set()
    weak_matched_skills=set()
    missing_skills_semantic=set()
    jd_list=list(jd_keywords)
    jd_emb=model.encode(jd_list,convert_to_tensor=True)
    res_emb=model.encode(resume_sentences,convert_to_tensor=True)

    for i,jd_kw in enumerate(jd_list):

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

    