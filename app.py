import streamlit as st
from modules.preprocessing import extract_text_from_pdf
from modules.analyzer import analyze_resume
from modules.model_utils import download_nltk_resources

download_nltk_resources()


st.set_page_config(page_title='Resume Matcher',layout='wide')
st.title('Resume Job Match Scorer')


col1,col2=st.columns(2)

with col1:

    st.subheader('Resume')

    uploaded_file=st.file_uploader('Upload your resume(PDF/TXT)',type=['pdf','txt'])

    resume_text=""
    if uploaded_file:
        if uploaded_file.type=='application/pdf':
            resume_text=extract_text_from_pdf(uploaded_file)

        else:
            resume_text=uploaded_file.read().decode('utf-8')


    st.write('OR')
    resume_text=st.text_area('Paste your resume',value=resume_text)

with col2:

    st.subheader('Job Description')
    jd_text=st.text_area('Paste Job Description')


#Analyze button

analyze=st.button('Analyze')

if analyze:
    if not resume_text or not jd_text:
        st.write('Please provide both inputs')
    
    else:
        with st.spinner('Analyzing...'):
            result=analyze_resume(jd_text,resume_text)

        st.subheader('Match Score')

        score=result['final_score']
        st.metric('Overall Match',f'{score}%')
        st.progress(int(score))
        
        #feedback colour
        if score>75:
            st.success('Strong Match')

        elif score>50:
            st.warning('Moderate Match')

        else:
            st.error('Low Match')


        #skills

        col3,col4=st.columns(2)

        with col3:

            st.subheader('Strong Skills')
            st.write(result['strong_skills'])

            st.subheader('Weak Skills')
            st.write(result['weak_skills'])

        with col4:

            st.subheader('Missing Skills')
            st.write(result['missing_skills'])


        st.subheader('Detailed Scores')
        st.write(f'TF-IDF Score: {result['tfidf']}')
        st.write(f'Semantic Score: {result['semantic']}')
        st.write(f'Skills Score: {result['skills_score']}')
        st.write(f'Sentences Score: {result['sentences_score']}')

        st.subheader('Explanation')
        st.write(result['explanation'])







    




