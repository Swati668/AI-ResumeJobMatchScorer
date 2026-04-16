
from agents.base_agent import BaseAgent
from agents.prompts import build_career_guidance_prompt
import json


class CareerGuidanceAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="career_guidance",
            llm_type=llm_type,
            temperature=0.6  
        )

    def build_prompt(self, context):
        return build_career_guidance_prompt(context)

    def postprocess(self, response, context):
        try:
            parsed = json.loads(response)
            return {
                "agent": self.name,
                "career_advice": parsed,
            }
        except:
            # fallback if JSON parsing fails
            return {
                "agent": self.name,
                "career_advice": response.strip(),
            }