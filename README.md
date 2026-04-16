# AI Resume Matcher

This project is a simple web app that helps compare a resume with a job description and tells how well they match.

I built this to understand how resumes are evaluated and to go beyond basic keyword matching by adding some AI-based analysis.



## What this project does

* You can upload a resume (PDF or TXT)
* Paste a job description
* The app compares both and shows:
  * Matching skills
  * Missing skills
  * Overall match score
* It also gives some AI-based feedback about the resume




## Why I call it an "AI Resume Matcher"

Most basic tools just check if keywords match.

This project tries to go a bit further:

* It uses **semantic similarity**, so it understands meaning, not just exact words
* It uses **LLMs (like Gemini or OpenAI)** to analyze the resume and give feedback
* It combines traditional ML (like TF-IDF) with AI-based reasoning

So instead of just saying “match/not match”, it explains *why*.



## Features

* Upload resume (PDF/TXT)
* Compare with job description
* Skill matching (matched + missing skills)
* TF-IDF based scoring
* Semantic similarity using transformers
* Categorization of missing skills
* AI-based resume analysis
* Ranking multiple resumes
  


## Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn (TF-IDF)
* Sentence Transformers
* FlashText, RapidFuzz
* Gemini



## How to run this project

### 1. Clone the repo


git clone <your-repo-link>
cd <your-project-folder>


### 2. Create virtual environment


python -m venv venv


Activate it:

* Mac/Linux:


source venv/bin/activate


* Windows:


venv\Scripts\activate


### 3. Install dependencies


pip install -r requirements.txt


### 4. (Optional) Set OpenAI / Gemini(Google) API key


export OPENAI_API_KEY=your_api_key


### 5. Run the app


streamlit run app.py



## Future improvements

* Better UI
* More detailed feedback
* Interview question suggestions



## Why I made this

I wanted to build something practical while learning:

* NLP
* Machine Learning
* How LLMs can be used in real applications

This project is a step towards building smarter AI tools for job preparation.

---
