from preprocess import preprocess_data
import pandas as pd
import numpy as np
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Normalizer


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
    df = pd.read_csv('data.csv')
    processed_df = preprocess_data(df)
    print(processed_df)


if __name__ == "__main__":
    main()
