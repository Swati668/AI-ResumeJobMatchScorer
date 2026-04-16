from agents.base_agent import BaseAgent
from agents.prompts import build_resume_reasoning_prompt
import json


class ResumeReasoningAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="resume_reasoning",
            llm_type=llm_type,
            temperature=0.4   # balanced reasoning
        )

    def build_prompt(self, context):
        return build_resume_reasoning_prompt(context)

    
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
                    "best_resume": parsed.get("best_resume"),
                    "reasoning": parsed.get("reasoning"),
                    "improvements": parsed.get("improvements")
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