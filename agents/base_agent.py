from llm.generate import generate_response
from cache import create_cache_key
import time
import json

agent_cache = {}


class BaseAgent:

    def __init__(self, name, llm_type="gemini", temperature=0.3, max_retries=2, debug=False):
        self.name = name
        self.llm_type = llm_type
        self.temperature = temperature
        self.max_retries = max_retries
        self.debug = debug

    
    def build_prompt(self, context):
        """
        Default prompt (override in child agents if needed)
        """
        resume = context.get("resume", "")
        jd = context.get("jd", "")

        return f"""
You are an expert AI resume evaluator.

Return only valid JSON.

Format:
{{
  "score": 0,
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

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""

   
    def _call_llm(self, prompt: str) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                return generate_response(
                    prompt=prompt,
                    llm_type=self.llm_type,
                    temperature=self.temperature
                )

            except Exception as e:
                if self.debug:
                    print(f"[{self.name}] Attempt {attempt+1} failed:", repr(e))
                time.sleep(1)

       
        return json.dumps({
            "error": True,
            "type": "GEMINI_FAILED",
            "message": "AI service failed after retries"
        })

    
    

    def postprocess(self, response, context):
        try:
            cleaned = response.strip()

            if "```" in cleaned:
                cleaned = cleaned.split("```")[-1].replace("json", "").strip()

            if not cleaned:
                raise ValueError("Empty response")

            parsed = json.loads(cleaned)

            
            if isinstance(parsed, dict) and parsed.get("error") is True:
                return {
                    "agent": self.name,
                    "status": "error",
                    "data": None,
                    "message": parsed.get("message", "AI analysis unavailable")
                }

            
            return {
                "agent": self.name,
                "status": "success",
                "data": parsed
            }

        except Exception:
            return {
                "agent": self.name,
                "status": "error",
                "data": None,
                "message": "Failed to parse AI response",
                "raw_output": response
            }

   
    def run(self, user_input: dict):

        cache_key = create_cache_key(self.name, user_input)

        if cache_key in agent_cache:
            return {**agent_cache[cache_key], "cached": True}

        prompt = self.build_prompt(user_input)
        response = self._call_llm(prompt)
        result = self.postprocess(response, user_input)

        agent_cache[cache_key] = result

        return {**result, "cached": False}