
def final_scoring(tfidf_score,semantic_score,skills_score,sentences_score):
    final_score=(
        0.3*tfidf_score
        + 0.3*semantic_score
        + 0.2*skills_score
        + 0.2*sentences_score
    )
    return round(final_score * 100, 2)

