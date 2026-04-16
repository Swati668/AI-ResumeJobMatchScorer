
# def build_resume_reviewer_prompt(data):
#     return f"""
# You are an expert Resume Reviewer and ATS Optimization Specialist.

# ### RULES:
# - Do NOT extract skills
# - Do NOT score the resume
# - ONLY rewrite bullet points
# - Do NOT change meaning
# - Do NOT add new information beyond reasonable metrics

# ### CONTEXT:
# Job Role: {data.get('job_role', '')}

# Weak Skills (for context only):
# {data.get('nlp_results', {}).get('weak_skills', [])}

# Missing Skills (for context only):
# {data.get('nlp_results', {}).get('missing_skills', [])}

# ### INPUT BULLETS:
# {data.get('resume_bullets', [])}

# ### TASK:
# Rewrite each bullet point to:
# - Start with strong action verbs
# - Be ATS optimized
# - Include measurable impact where possible
# - Be concise and professional

# ### OUTPUT:
# Return ONLY improved bullet points as a clean list.
# Do NOT include explanations.
# """

# def build_skill_gap_prompt(data):
#     return f"""
# You are a Skill Gap Analysis Expert.

# ### RULES:
# - Do NOT extract or detect skills
# - Do NOT compute any scores
# - ONLY interpret given data
# - Keep explanation simple and structured

# ### INPUT:
# Missing Skills: {data.get('missing_skills', [])}
# Weak Skills: {data.get('weak_skills', [])}
# Matched Skills: {data.get('matched_skills', [])}

# ### TASK:
# Provide:

# 1. Why these missing and weak skills matter in hiring
# 2. Impact on candidate selection chances
# 3. Priority order to fix skills (High → Low)
# 4. Short actionable advice

# ### OUTPUT FORMAT:
# - Clear headings
# - Bullet points
# - No unnecessary text
# """

# def build_skill_advisor_prompt(data):
#     return f"""
# You are a Career Skill Advisor.

# ### RULES:
# - Do NOT analyze or detect skills
# - ONLY use provided data
# - Focus on actionable learning guidance

# ### INPUT:
# Missing Skills: {data.get('missing_skills', [])}
# Weak Skills: {data.get('weak_skills', [])}

# ### TASK:
# For each missing skill, provide:

# 1. Learning path (beginner → advanced)
# 2. Best free or affordable resources
# 3. 1–2 practical project ideas
# 4. Estimated time to learn

# ### OUTPUT FORMAT:
# For each skill:
# - Skill Name
#   - Learning Path:
#   - Resources:
#   - Projects:
#   - Time Estimate:

# Keep it practical and structured.
# """

# def build_career_guidance_prompt(data):
#     return f"""
# You are a Senior Career Advisor.

# ### RULES:
# - Do NOT compute or infer new skills
# - ONLY use provided data
# - Focus on strategy and positioning

# ### INPUT:
# Job Role: {data.get('job_role', '')}

# Matched Skills:
# {data.get('nlp_results', {}).get('matched_skills', [])}

# Missing Skills:
# {data.get('nlp_results', {}).get('missing_skills', [])}

# Weak Skills:
# {data.get('nlp_results', {}).get('weak_skills', [])}

# ### TASK:
# Provide:

# 1. Best-fit job roles
# 2. Alternative career paths
# 3. Resume positioning strategy
# 4. Interview focus areas
# 5. 30-day improvement plan

# ### OUTPUT FORMAT:
# Use clear headings and bullet points.
# Be concise and practical.
# """

# def build_resume_formatter_prompt(data):
#     return f"""
# You are an ATS Resume Formatting Expert.

# ### RULES:
# - DO NOT rewrite content
# - DO NOT improve wording
# - DO NOT add new information
# - ONLY restructure the content

# ### INPUT RESUME:
# {data.get("improved_resume", "")}

# ### TASK:
# Convert the resume into structured ATS-friendly format.

# ### OUTPUT FORMAT (STRICT JSON):
# {{
#     "header": "",
#     "summary": "",
#     "skills": [],
#     "experience": [],
#     "projects": [],
#     "education": []
# }}

# ### FORMATTING RULES:
# - No tables or columns
# - No emojis or icons
# - Keep text clean and structured
# - Ensure all sections are properly filled

# Return ONLY valid JSON.
# """


def build_resume_reasoning_prompt(context):

      ranked_resumes = context.get("ranked_resumes", [])
      jd = context.get("job_description", "")

      return f"""
  You are an expert hiring manager.

  Resumes are already ranked using an NLP scoring system.

  ### Your job:
  1. Justify WHY the top-ranked resume is better than others
  2. Compare it with 2nd/3rd resume briefly# 3. Suggest improvements for the BEST resume

  ### Focus on:
  - Skill match with job description
  - Impact and measurable achievements
  - Relevance to role
  - Clarity and structure

  ### Output Format (STRICT JSON):
  {{
      "best_resume": "<resume_name_or_index>",
      "reasoning": "<clear explanation comparing top resumes>",
      "improvements": [        "<improvement 1>",
          "<improvement 2>",
          "<improvement 3>"
      ]
  }}

  ### Ranked Resumes (Top First):
  {ranked_resumes}

  ### Job Description:
  {jd}
  """

def build_combined_analysis_prompt(context):
    resume = context["resume"]
    jd = context["jd"]

    return f"""
You are an expert AI career assistant and resume evaluator.

Analyze the resume against the job description.

--- IMPORTANT ---
Return ONLY valid JSON. No explanations, no markdown.

--- OUTPUT FORMAT ---
{{
  "score": 85,
  "strengths": [],
  "weaknesses": [],
  "skill_gaps": [],
  "suggestions": [],
  "career_guidance": [],
  "ats_optimization_tips": [],
  "improved_resume": {{
    "summary": "",
    "experience_bullets": [],
    "skills_section": ""
  }}
}}

--- RULES ---
- Be strict and realistic with scoring
- Only use information from resume and JD
- Keep suggestions actionable
- Do NOT include extra text outside JSON

--- RESUME ---
{resume}

--- JOB DESCRIPTION ---
{jd}
"""