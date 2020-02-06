import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def tokenization_tweets(dataset, features):
    tokenization = TfidfVectorizer(max_features=features)
    tokenization.fit(dataset)
    dataset_transformed = tokenization.transform(dataset).toarray()
    return dataset_transformed


def main():
    df = pd.read_csv('data.csv')
    df = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(
            df['tweet'], df['sentiment'], test_size=0.2, shuffle=True)

    X_train_mod = tokenization_tweets(X_train, 3500)
    print(X_train_mod)


if __name__ == "__main__":
    main()
