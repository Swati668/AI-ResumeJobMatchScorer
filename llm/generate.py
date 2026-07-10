from llm.llm_loader import load_gemini_llm
import json


def generate_response(prompt: str,
                      llm_type: str = "gemini",
                      temperature: float = 0.3):
    
    client = load_gemini_llm()
   
    try:
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
       

        text = response.text

        if not text:
            raise RuntimeError("Empty Gemini response")

        text = text.strip()

        # Remove markdown code fences if Gemini returns them
        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        return text

    except Exception as e:
        import traceback

        print("=" * 80)
        print("GEMINI ERROR")
        print(type(e).__name__)
        print(str(e))
        traceback.print_exc()
        print("=" * 80)
        
        fallback = {
            "error": True,
            "type": "GEMINI_UNAVAILABLE",
            "message": (
                "AI analysis is temporarily unavailable. "
                "Please try again later."
            )
        }

        return json.dumps(fallback)