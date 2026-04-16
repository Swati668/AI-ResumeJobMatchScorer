from agents.legacy.resume_reviewer import ResumeReviewerAgent
from agents.legacy.skill_gap_analyzer import SkillGapAnalyzerAgent
from agents.legacy.skill_advisor import SkillAdvisorAgent
from agents.legacy.career_guidance import CareerGuidanceAgent
from agents.legacy.resume_formatter import ResumeFormatterAgent


class AgentOrchestrator:

    def __init__(self, llm_type="local"):
        self.reviewer = ResumeReviewerAgent(llm_type)
        self.gap = SkillGapAnalyzerAgent(llm_type)
        self.advisor = SkillAdvisorAgent(llm_type)
        self.career = CareerGuidanceAgent(llm_type)
        self.formatter = ResumeFormatterAgent(llm_type)

    def run_all(self, context: dict):
        """
        context = NLP output + resume + job role
        """

        results = {}

        # 1. Resume improvement
        results["review"] = self.reviewer.run(context)

        # 2. Skill gap analysis
        results["gap"] = self.gap.run(context)

        # 3. Skill roadmap
        results["advisor"] = self.advisor.run(context)

        # 4. Career guidance
        results["career"] = self.career.run(context)

        # 5. Formatting (imp: uses improved resume)
        formatter_input = {
            **context,
            "improved_resume": results["review"]["improved_resume"]
         }

        results["formatted"] = self.formatter.run(formatter_input)

        return results