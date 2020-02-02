from config import create_api
from preprocess import preprocess_data
from textblob import TextBlob
import tweepy
import re
import pandas as pd


class TweetStreamListner(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        self.me = api.me()

    def clean_tweet(self, tweet):
        return ' '.join(re.sub('''(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|
                        (\\w+:\\/\\/\\S+)''', '', tweet).split())

    def on_status(self, tweet):
        stream_tweet = self.clean_tweet(tweet.text)
        print(stream_tweet)

    def on_error(self, status_code):
        print('Error')


def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Netural'


def save_data(api):
    search = 'NRC -filter:retweets'
    searched_tweet = tweepy.Cursor(api.search, q=search).items(500)
    tweets_data = [[tweet.user.name, tweet.text]
                   for tweet in searched_tweet]
    df = pd.DataFrame(tweets_data, columns=['user', 'tweet'])
    df = df.drop_duplicates('tweet')
    processed_df = preprocess_data(df)
    processed_df['sentiment'] = processed_df['tweet'].apply(get_sentiment)
    processed_df = processed_df.drop_duplicates('tweet')
    processed_df = processed_df.dropna()
    processed_df.to_csv('data.csv', index=False)


def main():
    # Get twitter data
    api = create_api()
    save_data(api)

    # Streaming Data
    # tweet_listner = TweetStreamListner(api)
    # stream = tweepy.Stream(api.auth, tweet_listner)
    # # stream.userstream('BJP','CONGRESS')
    # stream.filter(track=['NRC'], languages=['en'])


if __name__ == "__main__":
    main()
