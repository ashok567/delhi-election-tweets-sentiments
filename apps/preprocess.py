import pandas as pd
import numpy as np
import re
# import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Normalizer


def handle_negation(row):
    # Tokenization
    words = word_tokenize(row)
    # POS
    # tags = nltk.pos_tag(words)
    return words


def replace_elongated_word(word):
    regex = r'(\w*)(\w+)\2(\w*)'
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    new_word = re.sub(regex, repl, word)
    if new_word != word:
        return replace_elongated_word(new_word)
    else:
        return new_word


def detect_elongated_words(row):
    regexrep = r'(\w*)(\w+)(\2)(\w*)'
    words = [''.join(i) for i in re.findall(regexrep, row)]
    exp_words = ['npr', 'nrc', 'caa', 'cab', 'bjp', 'congress']
    for word in words:
        if not wordnet.synsets(word) and word not in exp_words:
            row = re.sub(word, replace_elongated_word(word), row)
    return row


def preprocess(df):
    # Replace links, @UserNames, blank spaces, etc.
    df['tweet'] = df['tweet'].str.lower().replace('rt', '')
    df['user'] = df['user'].str.lower().replace(r'[^0-9A-Za-z \t]', '', regex=True)
    df['tweet'] = df['tweet'].str.lower().replace(r'[^0-9A-Za-z \t]', '', regex=True)
    df['tweet'] = df['tweet'].str.replace(r'@\w+', '', regex=True)
    df['tweet'] = df['tweet'].str.replace(r'http\S+', '', regex=True)
    df['tweet'] = df['tweet'].str.replace(r'www\S+', '', regex=True)
    df['tweet'] = df['tweet'].replace(r'[0-9]+', '', regex=True)
    df['tweet'] = df['tweet'].replace(
                  r'[!"$%&#()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True)
    # Replace elongated words by identifying those repeated characters and \
    # compare the new word with the english lexicon
    df['tweet'] = df['tweet'].apply(lambda x: detect_elongated_words(x))
    # df['tweet'] = df['tweet'].apply(handle_negation)
    # Stopwords
    stop_word_list = stopwords.words('english')
    df['tweet'] = df['tweet'].apply(
                  lambda x: ' '.join([word for word in x.split()
                                     if word not in stop_word_list]))
    return df


def vectorization(df):
    vector = CountVectorizer()
    freq_matrix = vector.fit_transform(df['tweet'])
    sum_freq = np.sum(freq_matrix, axis=0)
    frequency = np.squeeze(np.asarray(sum_freq))
    # print(vector.get_feature_names())
    # print(frequency)
    freq_df = pd.DataFrame([frequency], columns=vector.get_feature_names()).T
    return freq_df


def main():
    df = pd.read_csv('../data/data.csv')
    df = preprocess(df)


if __name__ == "__main__":
    main()
