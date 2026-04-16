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
            st.write("Step 1: Starting analysis")
            nlp_result = analyze_resume(jd_text, resume_text)

            st.write("Step 2: Got result")
            st.write(nlp_result)

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

        with col1:
            st.subheader("Strong Skills")
            st.success(" • ".join(nlp_result['strong_skills']))

            st.subheader("Weak Skills")
            st.warning(" • ".join(nlp_result['weak_skills']))

        with col2:
            st.subheader("Missing Skills")
            st.error(" • ".join(nlp_result['missing_skills']))

    # ---------------- TAB 4: SKILL IMPORTANCE ----------------
    with tab4:
        st.subheader("Skill Importance Analysis")

        missing_skills_importance = nlp_result['missing_skills_importance']

        critical, moderate, optional = [], [], []

        for skill, label in missing_skills_importance.items():
            if label == "critical":
                critical.append(skill)
            elif label == "optional":
                optional.append(skill)
            else:
                moderate.append(skill)

        st.markdown("**Critical Skills**")
        for skill in critical:
            context_text = best_context(skill, jd_text)
            st.error(f"{skill} → {context_text}")

        st.markdown("**Moderate Skills**")
        for skill in moderate:
            context_text = best_context(skill, jd_text)
            st.warning(f"{skill} → {context_text}")

        st.markdown("**Optional Skills**")
        for skill in optional:
            context_text = best_context(skill, jd_text)
            st.success(f"{skill} → {context_text}")

    # ---------------- TAB 5: DETAILED SCORES ----------------
    with tab5:
        st.subheader("Detailed Scores")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"TF-IDF Score: {nlp_result['tfidf']:.2f}")
            st.write(f"Semantic Score: {nlp_result['semantic']:.2f}")

        with col2:
            st.write(f"Skills Score: {nlp_result['skills_score']:.2f}")
            st.write(f"Sentence Score: {nlp_result['sentences_score']:.2f}")

        st.subheader('Section Wise Scores')

        for section, score_val in nlp_result['section_scores'].items():
            if score_val is None:
                st.write(f'{section.capitalize()} : N/A')
            else:
                st.write(f'{section.capitalize()}: {score_val:.2f}')

    
    # ---------------- TAB 6: INSIGHTS + AGENTS ----------------
    with tab6:

        st.subheader("AI Insights Dashboard")

    # ---------------- NLP INSIGHTS ----------------
        with st.expander("NLP Skill Insights", expanded=False):
            for item in nlp_result['explanation']:
                skill = item['skill']
                match = item['match_type']
                reason = item['reason']

                if match == "STRONG":
                    st.success(f"{skill} → {reason}")
                elif match == "WEAK":
                    st.warning(f"{skill} → {reason}")
                else:
                    st.error(f"{skill} → {reason}")

        st.divider()

        # ---------------- (COMBINED AGENT) ----------------
        if agent_result and "career_analyst" in agent_result:

            data = agent_result["career_analyst"]
            if agent_result['career_analyst']["status"] == "error":
                st.warning(agent_result['career_analyst']["message"])

            else:
        
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

                    improved = data.get("improved_resume", "")

                    if isinstance(improved, dict):
                        st.markdown("### Summary")
                        st.markdown(improved.get("summary", "No data"))

                        st.markdown("### Experience Bullets")
                        st.markdown("\n".join(improved.get("experience_bullets", [])))

                        st.markdown("### Skills Section")
                        st.markdown(improved.get("skills_section", "No data"))
                    else:
                        st.markdown(improved if improved else "No data")

            st.divider()

    # ---------------- PDF DOWNLOAD ----------------
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        pdf_file = "resume_report.pdf"
        doc = SimpleDocTemplate(pdf_file)
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph("Resume Match Report", styles['Title']))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"Overall Score: {score}%", styles['Normal']))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"Strong Skills: {', '.join(nlp_result['strong_skills'])}", styles['Normal']))
        content.append(Paragraph(f"Weak Skills: {', '.join(nlp_result['weak_skills'])}", styles['Normal']))
        content.append(Paragraph(f"Missing Skills: {', '.join(nlp_result['missing_skills'])}", styles['Normal']))

        content.append(Spacer(1, 10))

        content.append(Paragraph(f"TF-IDF Score: {nlp_result['tfidf']:.2f}", styles['Normal']))
        content.append(Paragraph(f"Semantic Score: {nlp_result['semantic']:.2f}", styles['Normal']))
        content.append(Paragraph(f"Skills Score: {nlp_result['skills_score']:.2f}", styles['Normal']))
        content.append(Paragraph(f"Sentence Score: {nlp_result['sentences_score']:.2f}", styles['Normal']))

        doc.build(content)

        with open(pdf_file, "rb") as f:
                st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="resume_report.pdf",
                mime="application/pdf"
            )

# ---------------- RANKING RESUMES ----------------
with tab7:
    st.header("Multi-Resume Ranking System (ATS Simulation)")

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

            summary_data = [
                {
                    "Rank": idx + 1,
                    "Resume": res["filename"],
                    "Score (%)": res["final_score"]
                }
                for idx, res in enumerate(results)
            ]

            st.dataframe(summary_data, use_container_width=True)

            st.divider()

            # ---------------- TOP RESUMES ----------------
            top_resume = results[0]

            st.subheader("Best Resume")

            st.success(f"{top_resume['filename']} — {top_resume['final_score']}%")

            # ---------------- DETAILED VIEW ----------------
            st.subheader("Detailed Breakdown")

            for idx, res in enumerate(results, 1):
                with st.expander(f"🏅 Rank {idx}: {res['filename']} ({res['final_score']}%)"):

                    if "error" in res:
                        st.error(f"Error processing file: {res['error']}")
                    else:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("🧠 Semantic", f"{res['semantic_score']}%")
                            st.metric("🛠 Skills", f"{res['skills_score']}%")

                        with col2:
                            st.metric("📄 Structure", f"{res['structure_score']}%")
                            st.metric("⭐ Final", f"{res['final_score']}%")

            # ---------------- REASONING AGENT ----------------
            if use_reasoning:

                st.divider()
                st.subheader("🧠 AI Resume Reasoning & Improvement")
                top_k = results[:3]

                ranked_resumes = [
                    {
                        "name": r["filename"],
                        "score": r["final_score"],
                        "summary": r.get("summary", "")[:500]  # limit text
                    }
                    for r in top_k
                ]

                from agents.resume_reasoning_agent import ResumeReasoningAgent

                reasoning_agent = ResumeReasoningAgent(llm_type="gemini")

                with st.spinner("Generating AI reasoning..."):
                    reasoning_result = reasoning_agent.run({
                        "ranked_resumes": ranked_resumes,
                        "job_description": jd_text
                    })

                
                # ---------------- DISPLAY ----------------
                if reasoning_result["status"] == "error":
                    st.warning("AI analysis not available due to API limit or service error")

                else:
                    data = reasoning_result["data"]

                    st.markdown("### Why this resume is best")
                    st.info(data.get("reasoning", "No reasoning available"))

                    st.markdown("### How to improve the best resume")

                    improvements = data.get("improvements", [])
                    if improvements:
                        for imp in improvements:
                            st.write(f"• {imp}")
                    else:
                        st.write("No improvements suggested")