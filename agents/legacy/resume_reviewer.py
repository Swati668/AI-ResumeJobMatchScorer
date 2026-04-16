from agents.base_agent import BaseAgent
from agents.prompts import build_resume_reviewer_prompt


class ResumeReviewerAgent(BaseAgent):

    def __init__(self, llm_type="gemini"):
        super().__init__(
            name="resume_reviewer",
            llm_type=llm_type,
            temperature=0.7#(temp. higher = better rewriting & creativity)
        )

    def build_prompt(self, context):
        return build_resume_reviewer_prompt(context)

    def postprocess(self, response, context):
        return {
            "agent": self.name,
            "improved_resume": response.strip()
        }