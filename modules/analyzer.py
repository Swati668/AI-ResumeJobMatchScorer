from modules.preprocessing import (clean_text_light,clean_text,get_sentences,extract_text_from_pdf)
from modules.utils import (tfidf_similarity,semantic_score,extract_sections,apply_fallbacks,section_wise_scores)
from modules.skills_extraction import extract_skills
from modules.matching import (fuzzy_matching,semantic_matching,semantic_sentence_matching,generate_explanation,assign_skill_importance)
from modules.scoring import final_scoring


def analyze_resume(jd_text,resume_text):
    clean_jd=clean_text(jd_text)
    clean_resume=clean_text(resume_text)

    light_clean_jd=clean_text_light(jd_text)
    light_clean_resume=clean_text_light(resume_text)

    jd_sentences=get_sentences(jd_text)
    resume_sentences=get_sentences(resume_text)

    jd_skills = extract_skills(jd_text)
    resume_skills=extract_skills(resume_text)


    fuzzy_matched_skills,fuzzy_missing_skills = fuzzy_matching(jd_skills,resume_skills,80)
    strong_matched_skills,weak_matched_skills,semantic_missing_skills = semantic_matching(jd_skills,
                                resume_skills,clean_resume.split(),0.6,0.3)
    strong_matched_sent,weak_matched_sent,missing_sent=semantic_sentence_matching(jd_sentences,
                                resume_sentences,0.6,0.3)
    missing_skills_importance=assign_skill_importance(jd_text,semantic_missing_skills)
    
    jd_sections=extract_sections(jd_text)
    
    
    resume_sections=extract_sections(resume_text)
    resume_sections=apply_fallbacks(resume_sections,resume_text)

    strong_keys={k for k,_ in strong_matched_skills}
    weak_keys={k for k,_ in weak_matched_skills}
    weak_match=weak_keys-strong_keys
    fuzzy_match=fuzzy_matched_skills-strong_keys-weak_match

    skills_section_score=((len(strong_keys)+len(weak_match)*0.5)/len(jd_skills)) if jd_skills else None



    tfidf_score=tfidf_similarity(clean_jd,clean_resume)
    semantic_score_value=semantic_score(light_clean_jd,light_clean_resume,resume_sentences,3)
    structure_score,section_scores=section_wise_scores(jd_sections,resume_sections,skills_section_score,3)
    skills_score=(1.0*(len(strong_keys)/len(jd_skills))
              + 0.6*(len(weak_match)/len(jd_skills))
              + 0.8*(len(fuzzy_match)/len(jd_skills))
              )
    skills_score=min(skills_score,1.0)

    sentences_score=(1.0*len(strong_matched_sent)
                + 0.5*len(weak_matched_sent))/len(jd_sentences) if jd_sentences else 0

    final=final_scoring(tfidf_score,semantic_score_value,skills_score,sentences_score,structure_score)

    explanation=generate_explanation(strong_matched_skills,weak_matched_skills,semantic_missing_skills)
    

    return {
        'final_score': final,
        "tfidf": tfidf_score,
        "semantic": semantic_score_value,
        "skills_score": skills_score,
        "sentences_score": sentences_score,
        "structure_score":structure_score,
        "section_scores":section_scores,
        "strong_skills": list(strong_keys),
        "weak_skills": list(weak_match),
        "missing_skills": list(semantic_missing_skills),
        "missing_skills_importance":missing_skills_importance,
        #"missing_skills": list(semantic_missing_skills) if isinstance(semantic_missing_skills, set) else semantic_missing_skills,
        "explanation": explanation
        
    }

