import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from twitter_data import load_data


def tokenization_tweets(dataset, features):
    tokenization = TfidfVectorizer(max_features=features)
    tokenization.fit(dataset)
    dataset_transformed = tokenization.transform(dataset).toarray()
    return dataset_transformed


def train(X_train_mod, y_train_mod, features, shuffle, drop, layer1, layer2, epoch, lr, epsilon, validation):
    model_nn = Sequential()
    model_nn.add(Dense(layer1, input_shape=(features,), activation='relu'))
    model_nn.add(Dropout(drop))
    model_nn.add(Dense(layer2, activation='sigmoid'))
    model_nn.add(Dropout(drop))
    model_nn.add(Dense(3, activation='softmax'))

    model_nn.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
    model_nn.fit(X_train_mod, y_train_mod, batch_size=32, epochs=epoch, verbose=1, validation_split=validation, shuffle=shuffle)
    return model_nn


def model1(X_train, y_train):
    features = 3500
    shuffle = True
    drop = 0.5
    layer1 = 512
    layer2 = 256
    epoch = 5
    lr = 0.005
    epsilon = 1e-8
    validation = 0.1
    X_train_mod = tokenization_tweets(X_train, features)
    y_train_mod = y_train.values.reshape((-1, 1))
    model = train(X_train_mod, y_train_mod, features, shuffle, drop, layer1, layer2, epoch, lr, epsilon, validation)
    return model


def main():
    load_data()
    df = pd.read_csv('data.csv')
    df = df.dropna()
    df['sentiment'] = df['sentiment'].apply(lambda x: 2 if x == 'Positive' else (0 if x == 'Negative' else 1))
    X_train, X_test, y_train, y_test = train_test_split(
            df['tweet'], df['sentiment'], test_size=0.2, shuffle=True)

    model = model1(X_train, y_train)
    X_test_mod = tokenization_tweets(X_test, 3500)
    predict = model.predict(X_test_mod)


if __name__ == "__main__":
    main()
