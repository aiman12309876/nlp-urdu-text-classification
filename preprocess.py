import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class UrduPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('urdu'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^آ-ی\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)

    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

    def preprocess_dataframe(self, df, text_col='text'):
        df['cleaned_text'] = df[text_col].apply(self.preprocess)
        return df