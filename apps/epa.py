import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Normalizer


def plot_graph(word_frequency):
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(50), word_frequency['count'][1:51], width = 0.8)
    plt.xticks(np.arange(50), word_frequency['count'][1:51].index, rotation=90)
    plt.xlabel('Frequent Words')
    plt.ylabel('Frequency Counts')
    plt.title('Word Frequency related to the subject')
    plt.show()


def vectorization(df):
    vector = CountVectorizer()
    freq_matrix = vector.fit_transform(df['tweet'])
    sum_freq = np.sum(freq_matrix, axis=0)
    frequency = np.squeeze(np.asarray(sum_freq))
    # print(vector.get_feature_names())
    # print(frequency)
    freq_df = pd.DataFrame([frequency], columns=vector.get_feature_names()).T
    freq_df.columns = ['count']
    return freq_df


def main():
    df = pd.read_csv('data.csv')
    df = df.dropna()
    word_frequency = vectorization(df).sort_values('count', ascending=False)
    plot_graph(word_frequency)


if __name__ == "__main__":
    main()
