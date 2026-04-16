from agents.base_agent import BaseAgent
from agents.prompts import build_skill_gap_prompt


class SkillGapAnalyzerAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="skill_gap_analyzer",
            llm_type=llm_type,
            temperature=0.3  
        )

    def build_prompt(self, context):
        return build_skill_gap_prompt(context)

    def postprocess(self, response, context):
        return {
            "agent": self.name,
            "skill_gap_report": response.strip()
        }