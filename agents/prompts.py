
def build_resume_reasoning_prompt(context):

      ranked_resumes = context.get("ranked_resumes", [])
      jd = context.get("job_description", "")

      return f"""
  You are an expert hiring manager.

  Resumes are already ranked using an NLP scoring system.

  ### Your job:
  If the score difference is less than 2%, mention that both resumes are similarly matched and explain the small differences instead of claiming one is clearly superior.
  Otherwise,
  1. Justify WHY the top-ranked resume is better than others
  2. Compare it with 2nd/3rd resume briefly
  3. Suggest improvements for the BEST resume

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