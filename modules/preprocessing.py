import nltk

def get_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


def nltk_tokenizers():
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    
nltk_tokenizers()


import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pdfplumber


def normalization(tokens):
    normalized=[]
    mapping={
        'ml':'machine learning',
        'ai':'artificial intelligence',
        'dl':'deep learning',
        'nlp':'natural language processing'

    }

    for token in tokens:
        if token in mapping:
            normalized.extend(mapping[token].split())
        else :
            normalized.append(token)
    return normalized


def get_sentences(text):
    return sent_tokenize(text)


stop_words = get_stopwords()
def clean_text(text):
    text=text.lower()

    text=text.translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
    text=text.replace('\n',' ')

    words=word_tokenize(text)
    normalized=normalization(words)

    custom_words=stop_words-{'c','r','go'}
    filtered_words=[word for word in normalized if word not in custom_words]

    return " ".join(filtered_words)


def extract_text_from_pdf(file_path):
    text=""

    with pdfplumber.open(file_path) as pdf:
         for page in pdf.pages:
             page_text=page.extract_text()

             if page_text:
                 text+=page_text+'\n'

    return text



