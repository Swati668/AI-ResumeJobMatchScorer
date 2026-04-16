from agents.base_agent import BaseAgent
from agents.prompts import build_combined_analysis_prompt
import json


class CareerAnalystAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="career_analyst",
            llm_type=llm_type,
            temperature=0.3   
        )

    def build_prompt(self, context):
        """
        context should contain:
        {
            "resume": "...",
            "jd": "..."
        }
        """
        return build_combined_analysis_prompt(context)


    def postprocess(self, response, context):
        try:
            parsed = json.loads(response)
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
                "data": {
                    "score": parsed.get("score", 0),
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", []),
                    "skill_gaps": parsed.get("skill_gaps", []),
                    "suggestions": parsed.get("suggestions", []),
                    "career_guidance": parsed.get("career_guidance", []),
                    "improved_resume": parsed.get("improved_resume", "")
                }
            }

        except Exception:
            return {
                "agent": self.name,
                "status": "error",
                "data": None,
                "message": "Failed to parse AI response",
                "raw_output": response.strip()
            }
        
    