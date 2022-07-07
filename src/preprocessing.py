import string
import nltk
from nltk.corpus import stopwords

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def remove_stopwords(text):
    nltk.download('stopwords')
    stopword = stopwords.words('russian')
    words = text.split()
    ans = ''
    for word in words:
        if word not in stopword:
            ans += word + ' '
    
    return ans

def preprocess(df, data_name: str, label_name: str):
    df[label_name] = df[label_name].astype('int')
    df[data_name] = df[data_name].apply(lambda x: remove_punctuation(x))
    df[data_name]=df[data_name].apply(lambda x: x.lower()) 
    df[data_name] = df[data_name].apply(lambda x: remove_stopwords(x))

    return df

    