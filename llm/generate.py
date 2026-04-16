from llm.llm_loader import load_gemini_llm
import json

def generate_response(prompt: str, llm_type: str = "gemini", temperature: float = 0.3):

    model = load_gemini_llm()

    try:
        response = model.generate_content(prompt)
        text = ""

        if response:
            if hasattr(response, "text") and response.text:
                text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                text = response.candidates[0].content.parts[0].text

        if not text:
            raise RuntimeError("Empty Gemini response")

        return text.strip()

    except Exception as e:
        print("GEMINI FAILED:", repr(e))
        fallback = {
            "error": True,
            "type": "GEMINI_UNAVAILABLE",
            "message": "AI analysis not available due to API limit or service error"
        }

        return json.dumps(fallback)