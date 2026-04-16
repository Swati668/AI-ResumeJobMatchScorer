
def final_scoring(tfidf_score,semantic_score,skills_score,sentences_score,structure_score):
    final_score=(
        0.25*tfidf_score
        + 0.35*semantic_score
        + 0.20*skills_score
        + 0.10*sentences_score
        + 0.10*structure_score
    )
    return round(final_score * 100, 2)

