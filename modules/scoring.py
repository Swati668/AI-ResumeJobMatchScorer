
def final_scoring(tfidf_score, semantic_score, skills_score,
                  sentences_score, structure_score):

    final_score = (
        0.10 * tfidf_score +       # Keyword overlap
        0.30 * semantic_score +    # Context similarity
        0.35 * skills_score +      # Skill matching 
        0.10 * sentences_score +   # Sentence-level relevance
        0.15 * structure_score     # Resume completeness
    )

    return round(final_score * 100, 2)