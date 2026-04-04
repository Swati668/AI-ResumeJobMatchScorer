from modules.preprocessing import (clean_text,get_sentences,extract_text_from_pdf)
from modules.utils import (tfidf_similarity,semantic_score,finding_keywords,generate_explanation)
from modules.skills_extraction import extract_skills
from modules.matching import (fuzzy_matching,semantic_matching,semantic_sentence_matching)
from modules.scoring import final_scoring


def analyze_resume(jd_text,resume_text):
    clean_jd=clean_text(jd_text)
    clean_resume=clean_text(resume_text)

    jd_sentences=get_sentences(jd_text)
    resume_sentences=get_sentences(resume_text)

    jd_skills = extract_skills(jd_text)
    resume_skills=extract_skills(resume_text)


    fuzzy_matched_skills,fuzzy_missing_skills = fuzzy_matching(jd_skills,resume_skills,80)
    strong_matched_skills,weak_matched_skills,semantic_missing_skills = semantic_matching(jd_skills,
                                resume_skills,clean_resume.split(),0.6,0.3)
    strong_matched_sent,weak_matched_sent,missing_sent=semantic_sentence_matching(jd_sentences,
                                resume_sentences,0.6,0.3)


    strong_keys={k for k,_ in strong_matched_skills}
    weak_keys={k for k,_ in weak_matched_skills}
    weak_match=weak_keys-strong_keys
    fuzzy_match=fuzzy_matched_skills-strong_keys-weak_match



    tfidf_score=tfidf_similarity(clean_jd,clean_resume)
    semantic_score_value=semantic_score(clean_jd,clean_resume,resume_sentences)
    skills_score=(1.0*(len(strong_keys)/len(jd_skills))
              + 0.6*(len(weak_match)/len(jd_skills))
              + 0.8*(len(fuzzy_match)/len(jd_skills))
              )

    sentences_score=(1.0*len(strong_matched_sent)
                + 0.5*len(weak_matched_sent))/len(jd_sentences) if jd_sentences else 0

    final=final_scoring(tfidf_score,semantic_score_value,skills_score,sentences_score)

    explanation=generate_explanation(strong_matched_skills,weak_matched_skills,semantic_missing_skills)
    

    return {
        'final_score': final,
        "tfidf": tfidf_score,
        "semantic": semantic_score_value,
        "skills_score": skills_score,
        "sentences_score": sentences_score,
        "strong_skills": list(strong_keys),
        "weak_skills": list(weak_match),
        "missing_skills": list(semantic_missing_skills),
        #"missing_skills": list(semantic_missing_skills) if isinstance(semantic_missing_skills, set) else semantic_missing_skills,
        "explanation": explanation
    }

