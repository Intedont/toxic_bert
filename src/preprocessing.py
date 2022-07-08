import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_punctuation(text):
    '''Функция получает на вход предложение, удаляет пунктуацию и возвращает обратно'''
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def remove_stopwords(text):
    '''Функция получает на вход предложение, удаляет стоп слова и возвращает его обратно'''
    stopword = stopwords.words('russian')
    words = text.split()
    ans = ''
    for word in words:
        if word not in stopword:
            ans += word + ' '
    
    return ans

def preprocess(source_df, data_name: str, label_name: str):
    '''Функция выполняет минимальную предобработку входного датафрейма
    Данные в столбце ответов label_name приводятся к типу Int
    Из колонки с данными data_name удаляются знаки пунктуации, стоп слова, а также колонка приводится к нижнему регистру
    Функция возвращает копию датафрейма
    '''
    df = source_df.copy()
    df[label_name] = df[label_name].astype('int')
    df[data_name] = df[data_name].apply(lambda x: remove_punctuation(x))
    df[data_name]= df[data_name].apply(lambda x: x.lower()) 
    df[data_name] = df[data_name].apply(lambda x: remove_stopwords(x))

    return df

    