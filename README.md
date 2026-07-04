# AI Resume Matcher

This project is a web application that compares resumes with job descriptions and estimates how well they match.

It goes beyond simple keyword matching by combining traditional NLP techniques with semantic similarity and LLM-based feedback to provide more meaningful resume analysis.

## Live Demo

**Streamlit Dashboard:**  
https://ai-resumejobmatchscorer-bwm3pferiappw24ewfaffx7.streamlit.app

---

## What this project does

- Upload one or more resumes (PDF or TXT)
- Paste a job description
- Compare resumes with the job description
- View:
  - Overall match score
  - Matched skills
  - Missing skills
- Generate AI-based resume feedback
- Rank multiple resumes based on their relevance

---

## Why this is an AI Resume Matcher

Most basic resume screening tools rely only on keyword matching.

This project combines multiple approaches to provide a more meaningful evaluation:

- Uses **TF-IDF** for keyword-based similarity
- Uses **Sentence Transformers** for semantic similarity
- Uses **Gemini** to generate personalized resume feedback
- Combines traditional NLP techniques with LLM-based reasoning

Instead of simply indicating whether a resume matches a job description, the application explains **why** and highlights areas for improvement.

---

## Features

- Resume upload (PDF/TXT)
- Job description matching
- Match score generation
- Matched and missing skills analysis
- TF-IDF similarity scoring
- Semantic similarity using Sentence Transformers
- AI-generated resume feedback
- **Multi-Resume Ranking (AI-Assisted Ranking)**

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- FlashText
- RapidFuzz
- Gemini API

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <your-project-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

```bash
export GEMINI_API_KEY=your_api_key
```

*(On Windows, set it using the appropriate environment variable command.)*

### 5. Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

- Improve the UI/UX
- Support additional resume formats
- Generate interview questions based on the job description
- Add recruiter analytics and resume comparison insights

---

## Motivation

I built this project while learning NLP and Large Language Models to understand how AI can be applied to practical hiring workflows.

The goal was to combine traditional machine learning techniques with modern LLMs to create a smarter and more informative resume screening application.