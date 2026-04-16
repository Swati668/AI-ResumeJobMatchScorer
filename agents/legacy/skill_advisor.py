from agents.base_agent import BaseAgent
from agents.prompts import build_skill_advisor_prompt


class SkillAdvisorAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="skill_advisor",
            llm_type=llm_type,
            temperature=0.8   
        )

    def build_prompt(self, context):
        return build_skill_advisor_prompt(context)

    def postprocess(self, response, context):
        return {
            "agent": self.name,
            "skill_roadmap": response.strip()
        }