import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
import os
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_combined_df_split
# from sklearn.preprocessing import Normalizer


def plot_bar(word_frequency):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(50), word_frequency['count'][1:51], width=0.8)
    plt.xticks(np.arange(50), word_frequency['count'][1:51].index, rotation=90)
    plt.xlabel('Frequent Words')
    plt.ylabel('Frequency Counts')
    plt.title('Word Frequency related to the subject')
    plt.show()


def plot_graph(df):
    plt.figure(figsize=(6, 6))
    pts = plt.scatter(df['positive_count'], df['negative_count'],
                      c=df["positive_count"], cmap="bwr")
    plt.colorbar(pts)
    # sns.regplot(x='positive_count', y='negative_count', data=df,
    #             scatter=False, fit_reg=False)
    plt.xlabel('Positive Word Frequency')
    plt.ylabel('Negative Word Frequency')
    plt.title("Word frequency in Positive vs. Negative Tweets")
    plt.show()


def word_cloud(wc_text):
    file = os.getcwd()
    mask = np.array(Image.open(os.path.join(file, "delhi.jpg")))
    wc = WordCloud(max_words=100, mask=mask, background_color='white',
                   contour_width=1, contour_color="black",
                   colormap="nipy_spectral", stopwords=['nrc'],
                   width=2000, height=1000, max_font_size=100)
    wc.generate(wc_text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="hermite")
    plt.axis('off')
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
    word_frequency_pos = vectorization(df[df['sentiment'] == 'Positive']).sort_values('count', ascending=False)
    word_frequency_neg = vectorization(df[df['sentiment'] == 'Negative']).sort_values('count', ascending=False)
    plot_bar(word_frequency)
    combined_df = pd.concat([word_frequency_pos, word_frequency_neg], axis=1, sort=False)
    combined_df = combined_df.fillna(0)
    combined_df.columns = ['positive_count', 'negative_count']
    plot_graph(combined_df)
    wc_text = df['tweet'].to_string(index=False)
    word_cloud(wc_text)


if __name__ == "__main__":
    main()
