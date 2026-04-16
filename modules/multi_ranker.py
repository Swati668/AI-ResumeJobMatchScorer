from modules.preprocessing import extract_text_from_pdf
from modules.analyzer import analyze_resume

def rank_resumes(resume_files, jd_text):
    results = []

    for file in resume_files:
        try:
            resume_text = extract_text_from_pdf(file)

            output = analyze_resume(jd_text, resume_text)

            final_score = output.get('final_score', 0)
            semantic_score = output.get('semantic_score_value', 0)
            skills_score = output.get('skills_score', 0)
            structure_score = output.get('structure_score', 0)

            results.append({
                "filename": file.name,
                "final_score": final_score,
                "semantic_score": round(semantic_score * 100, 2),
                "skills_score": round(skills_score * 100, 2),
                "structure_score": round(structure_score * 100, 2)
            })

        except Exception as e:
            results.append({
                "filename": file.name,
                "error": str(e),
                "final_score": 0
            })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return results