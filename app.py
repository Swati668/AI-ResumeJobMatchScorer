import streamlit as st
from modules.preprocessing import extract_text_from_pdf
from modules.analyzer import analyze_resume
from modules.matching import best_context
from modules.multi_ranker import rank_resumes




st.set_page_config(page_title='AI Resume Matcher', layout='wide')

st.title('AI Resume Matcher')
st.caption('Match your resume with job descriptions intelligently')

st.divider()


tab1, tab2, tab3, tab4, tab5 , tab6 , tab7 = st.tabs([
    "Input",
    "Match Score",
    "Skills",
    "Skills Importance",
    "Detailed Scores",
    "Insights & Download",
    "Ranking Resumes"
])
 # ---------------- TAB 1: INPUT ----------------

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Resume')
        uploaded_file = st.file_uploader('Upload Resume (PDF/TXT)', type=['pdf', 'txt'])

        resume_text = ""
        if uploaded_file:
            if uploaded_file.type == 'application/pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode('utf-8')

        st.write('OR')
        resume_text = st.text_area('Paste Resume', value=resume_text, height=250)

    with col2:
        st.subheader('Job Description')
        jd_text = st.text_area('Paste Job Description', height=250)

        job_role = st.text_input('Job Role (Optional)')

    analyze = st.button('Analyze Resume')


# ---------------- ANALYSIS ----------------
if analyze:

    if not resume_text or not jd_text:
        st.error('Please provide both Resume and Job Description')
        st.stop()

   

    with st.spinner('Analyzing Resume + Running AI Agents...'):
        try:
            
            nlp_result = analyze_resume(jd_text, resume_text)

            from agents.combined_agent import CareerAnalystAgent
            
            career_analyst = CareerAnalystAgent()
            
            agent_output = career_analyst.run({
            "resume": resume_text,
            "jd": jd_text
            })

            agent_result = {
            "career_analyst": agent_output
                } 
            
        except Exception as e:
            st.error(f"Model failed: {e}")
            nlp_result = {}
            agent_result = {}


    score = nlp_result.get('final_score', 0)



    # ---------------- TAB 2: MATCH SCORE ----------------
    with tab2:
        st.subheader("Match Score")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Match", f"{score}%")

        with col2:
            if score > 75:
                st.success("Strong Match")
            elif score > 50:
                st.warning("Moderate Match")
            else:
                st.error("Low Match")

        st.progress(min(score / 100, 1.0))

    # ---------------- TAB 3: SKILLS ----------------
    with tab3:
        col1, col2 = st.columns(2)
        strong_skills = nlp_result.get("strong_skills", [])
        weak_skills = nlp_result.get("weak_skills", [])
        missing_skills = nlp_result.get("missing_skills", [])

        
        with col1:
            st.subheader("Strong Skills")
            st.success(" • ".join(strong_skills) if strong_skills else "No strong skills found.")

            st.subheader("Weak Skills")
            st.warning(" • ".join(weak_skills) if weak_skills else "No weak skills found.")

        with col2:
            st.subheader("Missing Skills")
            st.error(" • ".join(missing_skills) if missing_skills else "No missing skills found.")

    # ---------------- TAB 4: SKILL IMPORTANCE ----------------
    with tab4:
        st.subheader("Skill Importance Analysis")

        missing_skills_importance = nlp_result.get("missing_skills_importance", {})

        if not missing_skills_importance:
            st.info("No missing skills identified.")
        else:

            critical, moderate, optional = [], [], []

            for skill, label in missing_skills_importance.items():
                if label == "critical":
                    critical.append(skill)
                elif label == "optional":
                    optional.append(skill)
                else:
                    moderate.append(skill)

            if critical:
                st.markdown("### Critical Skills")
                for skill in critical:
                    context_text = best_context(skill, jd_text)
                    st.error(f"{skill} → {context_text}")

            if moderate:
                st.markdown("### Moderate Skills")
                for skill in moderate:
                    context_text = best_context(skill, jd_text)
                    st.warning(f"{skill} → {context_text}")

            if optional:
                st.markdown("### Optional Skills")
                for skill in optional:
                    context_text = best_context(skill, jd_text)
                    st.success(f"{skill} → {context_text}")


    # ---------------- TAB 5: DETAILED SCORES ----------------
    with tab5:

        st.subheader("Detailed Scores")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "TF-IDF Score",
                f"{nlp_result.get('tfidf', 0):.2f}"
            )

            st.metric(
                "Semantic Score",
                f"{nlp_result.get('semantic', 0):.2f}"
            )

        with col2:
            st.metric(
                "Skills Score",
                f"{nlp_result.get('skills_score', 0):.2f}"
            )

            st.metric(
                "Sentence Score",
                f"{nlp_result.get('sentences_score', 0):.2f}"
            )

        st.divider()

        st.subheader("Section-wise Scores")

        section_scores = nlp_result.get("section_scores", {})

        if not section_scores:
            st.info("Section-wise analysis is unavailable.")
        else:

            for section, score_val in section_scores.items():

                if score_val is None:
                    st.write(
                        f"**{section.capitalize()}** : "
                        "Not applicable (section not found in resume or job description)"
                    )
                else:
                    st.write(
                        f"**{section.capitalize()}** : "
                        f"{score_val:.2f}"
                )
    # ---------------- TAB 6: INSIGHTS + AGENTS ----------------
    with tab6:

        st.subheader("AI Insights Dashboard")

    # ---------------- NLP INSIGHTS ----------------
        with st.expander("NLP Skill Insights", expanded=False):
            explanations = nlp_result.get("explanation", [])

            if not explanations:
                st.info("No NLP insights available.")
            else:
                for item in explanations:
                    skill = item.get("skill", "")
                    match = item.get("match_type", "")
                    reason = item.get("reason", "")

                    if match == "STRONG":
                        st.success(f"{skill} → {reason}")
                    elif match == "WEAK":
                        st.warning(f"{skill} → {reason}")
                    else:
                        st.error(f"{skill} → {reason}")

        st.divider()

        # ---------------- (COMBINED AGENT) ----------------
        if agent_result and "career_analyst" in agent_result:

            
            response = agent_result["career_analyst"]

            if response["status"] == "error":
                st.warning(response["message"])
            else:
                data = response["data"]
            
                st.subheader("Resume Score")
                st.metric("Overall Score", data.get("score", "N/A"))
 
                with st.expander("Strengths"):
                    st.markdown("\n".join(data.get("strengths", ["No data"])))

                
                with st.expander("Weaknesses"):
                    st.markdown("\n".join(data.get("weaknesses", ["No data"])))

                
                with st.expander("Skill Gaps"):
                    st.markdown("\n".join(data.get("skill_gaps", ["No data"])))

                
                with st.expander("Suggestions to Improve"):
                    st.markdown("\n".join(data.get("suggestions", ["No data"])))

                
                with st.expander("Career Guidance"):
                    st.markdown("\n".join(data.get("career_guidance", ["No data"])))

                
                if "ats_optimization_tips" in data:
                    with st.expander("ATS Optimization Tips"):
                        st.markdown("\n".join(data.get("ats_optimization_tips", ["No data"])))

                # ---------------- IMPROVED RESUME ----------------
                with st.expander("Improved Resume (AI Version)", expanded=True):

                    improved = data.get("improved_resume", {})

                    if isinstance(improved, dict):

                        # Summary
                        st.markdown("### Summary")
                        st.markdown(improved.get("summary", "No data"))

                        # Experience
                        st.markdown("### Experience")

                        experience = improved.get("experience_bullets", [])

                        if not experience:
                            st.write("No experience generated.")

                        else:
                            for item in experience:

                                # Case 1: Gemini returned structured objects
                                if isinstance(item, dict):

                                    st.markdown(f"#### {item.get('title', 'Experience')}")

                                    tech = item.get("technologies")
                                    if tech:
                                        st.write(f"**Technologies:** {tech}")

                                    bullets = item.get("bullets", [])

                                    if isinstance(bullets, list):
                                        for bullet in bullets:
                                            st.write(f"• {bullet}")
                                    elif bullets:
                                        st.write(f"• {bullets}")

                                # Case 2: Gemini returned plain strings
                                else:
                                    st.write(f"• {item}")

                        # Skills Section
                        st.markdown("### Skills Section")
                        st.markdown(improved.get("skills_section", "No data"))

                    else:
                        st.markdown(improved if improved else "No data")

            st.divider()

    # ---------------- PDF DOWNLOAD ----------------
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from datetime import datetime

        pdf_file = "resume_report.pdf"

        doc = SimpleDocTemplate(pdf_file)
        styles = getSampleStyleSheet()

        content = []

                        # ---------------- Title ----------------
        content.append(Paragraph("AI Resume Match Report", styles["Title"]))
        content.append(Spacer(1, 12))

        content.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                styles["Normal"],
            )
        )

        content.append(Spacer(1, 15))

        # ---------------- Overall Score ----------------
        content.append(
            Paragraph(
                f"<b>Overall Match Score:</b> {score:.1f}%",
                styles["Normal"],
            )
        )

        content.append(Spacer(1, 12))

        # ---------------- Skills ----------------
        strong_skills = nlp_result.get("strong_skills", [])
        weak_skills = nlp_result.get("weak_skills", [])
        missing_skills = nlp_result.get("missing_skills", [])

        content.append(
            Paragraph(
                f"<b>Strong Skills:</b> "
                f"{', '.join(strong_skills) if strong_skills else 'None'}",
                styles["Normal"],
            )
        )

        content.append(
            Paragraph(
                f"<b>Weak Skills:</b> "
                f"{', '.join(weak_skills) if weak_skills else 'None'}",
                styles["Normal"],
            )
        )

        content.append(
            Paragraph(
                f"<b>Missing Skills:</b> "
                f"{', '.join(missing_skills) if missing_skills else 'None'}",
                styles["Normal"],
            )
        )

        content.append(Spacer(1, 12))

        # ---------------- Detailed Scores ----------------
        content.append(
            Paragraph(
                f"<b>TF-IDF Score:</b> {nlp_result.get('tfidf', 0):.2f}",
                styles["Normal"],
            )
        )

        content.append(
            Paragraph(
                f"<b>Semantic Score:</b> {nlp_result.get('semantic', 0):.2f}",
                styles["Normal"],
            )
        )

        content.append(
            Paragraph(
                f"<b>Skills Score:</b> {nlp_result.get('skills_score', 0):.2f}",
                styles["Normal"],
            )
        )

        content.append(
            Paragraph(
                f"<b>Sentence Score:</b> {nlp_result.get('sentences_score', 0):.2f}",
                styles["Normal"],
            )
        )

        doc.build(content)

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="resume_report.pdf",
                mime="application/pdf",
            )

# ---------------- RANKING RESUMES ----------------
with tab7:
    st.header("Multi-Resume Ranking System")

    st.markdown("Upload multiple resumes and rank them for a given job description.")

  
    jd_text = st.text_area(
        "Paste Job Description",
        height=200,
        placeholder="Paste job description here..."
    )

    uploaded_resumes = st.file_uploader(
        "Upload Multiple Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    
    use_reasoning = st.checkbox("Generate AI Reasoning & Improvements (Optional)")

    if st.button("Rank Resumes"):

        if not jd_text:
            st.warning("Please enter job description")

        elif not uploaded_resumes:
            st.warning("Please upload at least one resume")

        else:
            with st.spinner("Analyzing and ranking resumes..."):
                results = rank_resumes(uploaded_resumes, jd_text)
            
            st.success("Ranking Completed!")

            # ---------------- SUMMARY TABLE ----------------
            st.subheader("Ranking Summary")

            if not results:
                st.error("No resumes could be analyzed.")
                st.stop()

            summary_data = [
                {
                    "Rank": idx + 1,
                    "Resume": res.get("filename", "Unknown"),
                    "Score (%)": res.get("final_score", 0)
                }
                for idx, res in enumerate(results)
            ]

            st.dataframe(summary_data, use_container_width=True)

            st.divider()

            # ---------------- TOP RESUME ----------------
            top_resume = results[0]

            st.subheader("Best Resume")
            st.success(
                f"{top_resume.get('filename', 'Unknown')} — "
                f"{top_resume.get('final_score', 0)}%"
            )

            # ---------------- DETAILED BREAKDOWN ----------------
            st.subheader("Detailed Breakdown")

            for idx, res in enumerate(results, start=1):

                with st.expander(
                    f"Rank {idx}: "
                    f"{res.get('filename', 'Unknown')} "
                    f"({res.get('final_score', 0)}%)"
                ):

                    if res.get("error"):
                        st.error(res["error"])
                        continue

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Semantic",
                            f"{res.get('semantic_score', 0)}%"
                        )

                        st.metric(
                            "Skills",
                            f"{res.get('skills_score', 0)}%"
                        )

                    with col2:
                        st.metric(
                            "Structure",
                            f"{res.get('structure_score', 0)}%"
                        )

                        st.metric(
                            "Final",
                            f"{res.get('final_score', 0)}%"
                        )

            # ---------------- AI REASONING ----------------
            if use_reasoning:

                st.divider()
                st.subheader("AI Resume Reasoning & Improvement")

                top_k = results[:3]

                ranked_resumes = [
                    {
                        "name": r.get("filename", "Unknown Resume"),
                        "score": r.get("final_score", 0),
                        "missing_skills": r.get("missing_skills", []),
                        "strong_skills": r.get("strong_skills", []),
                        "weak_skills": r.get("weak_skills", []),

                        "resume_text": r.get("resume_text", "")[:1500],
                    }
                    for r in top_k
                ]

                from agents.resume_reasoning_agent import ResumeReasoningAgent

                reasoning_agent = ResumeReasoningAgent()

                with st.spinner("Generating AI reasoning..."):

                    reasoning_result = reasoning_agent.run(
                        {
                            "ranked_resumes": ranked_resumes,
                            "job_description": jd_text
                        }
                    )

                
                if reasoning_result.get("status") == "error":

                    st.info(
                        "AI reasoning is temporarily unavailable.\n\n"
                        "The resume ranking above is still generated using the NLP scoring engine."
                    )

                else:

                    data = reasoning_result.get("data", {})

                    st.markdown("### Why this resume is the best match")

                    st.info(
                        data.get(
                            "reasoning",
                            "No AI reasoning available."
                        )
                    )

                    st.markdown("### Suggested Improvements")

                    improvements = data.get("improvements", [])

                    if improvements:

                        for imp in improvements:
                            st.write(f"• {imp}")

                    else:
                        st.write("No improvement suggestions available.")