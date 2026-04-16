from agents.base_agent import BaseAgent
from agents.prompts import build_resume_formatter_prompt
import json


class ResumeFormatterAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="resume_formatter",
            llm_type=llm_type,
            temperature=0.2   
        )

    def build_prompt(self, context):
        return build_resume_formatter_prompt(context)

    def postprocess(self, response, context):
        try:
            parsed = json.loads(response)
            return {
                "agent": self.name,
                "formatted_resume": parsed
            }
        except:
            # fallback if model doesn't return JSON
            return {
                "agent": self.name,
                "formatted_resume": response.strip()
            }