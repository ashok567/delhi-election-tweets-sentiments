import re
import nltk
nltk.download("stopwords")
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


def replace_antonyms(word):
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                return lemma.antonyms()[0].name()
    return word


def handle_negation(row):
    # Tokenization
    words = word_tokenize(row)
    # POS
    tags = nltk.pos_tag(words)
    speech_tags = ['JJ', 'JJR', 'JJS', 'NN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP']
    neg = [words.index(i) for i in ['not', 'n\'t', 'nt'] if i in words]
    tags2 = ''
    if neg:
        tags2 = tags[min(neg):]
        words2 = words[min(neg):]
        words = words[:min(neg)+1]

    for index, word_tag in enumerate(tags2):
        if word_tag[1] in speech_tags:
            words = words+[replace_antonyms(word_tag[0])]+words2[index+2:]
    sent = ' '.join(words)
    return sent


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
    exp_words = ['npr', 'nrc', 'caa', 'cab', 'bjp', 'congress', 'aap']
    for word in words:
        if not wordnet.synsets(word) and word not in exp_words:
            row = re.sub(word, replace_elongated_word(word), row)
    return row


def clean_data(df):
    # Replace links, @UserNames, blank spaces, emoji etc.
    df['tweet'] = df['tweet'].str.lower().replace('rt', '')
    df['tweet'] = df['tweet'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['tweet'] = df['tweet'].str.replace(r'[^A-Za-z \t]', '', regex=True)
    df['tweet'] = df['tweet'].str.replace(r'@\w+', '', regex=True)
    df['tweet'] = df['tweet'].str.replace(r'http\S+', '', regex=True)
    df['tweet'] = df['tweet'].str.replace(r'www\S+', '', regex=True)
    df['tweet'] = df['tweet'].replace(r'[0-9]+', '', regex=True)
    df['tweet'] = df['tweet'].replace(
                  r'[!"$%&#\'()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True)

    # Replace elongated words by identifying those repeated characters and \
    # compare the new word with the english lexicon
    df['tweet'] = df['tweet'].apply(lambda x: detect_elongated_words(x))
    df['tweet'] = df['tweet'].apply(handle_negation)

    # Stopwords - English & Hinglish
    stopword_list1 = stopwords.words('english')
    stopwords_df = pd.read_csv('stopwords.csv')
    stopword_list2 = stopwords_df['words'].values.tolist()
    stopword_list = stopword_list1 + stopword_list2
    df['tweet'] = df['tweet'].apply(
                  lambda x: ' '.join([word for word in x.split()
                                     if word not in stopword_list]))
    df = df.dropna()
    return df


def preprocess_data(df):
    df = df.dropna()
    df = clean_data(df)
    return df
