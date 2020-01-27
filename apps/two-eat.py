from config import create_api
import tweepy
from textblob import TextBlob
import re
# import pandas as pd


class TweetStreamListner(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        self.me = api.me()

    def clean_tweet(self, tweet):
        return ' '.join(re.sub('''(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|
                        (\\w+:\\/\\/\\S+)''', '', tweet).split())

    def on_status(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet.text))
        if analysis.sentiment.polarity >= 0.5:
            print(tweet.text+': Subjective')
        elif analysis.sentiment.polarity < 0.5:
            print(tweet.text+': Objective')

    def on_error(self, status_code):
        print('Error')


def main():
    api = create_api()
    # search = 'CAA -filter:retweets'
    # searched_tweet = tweepy.Cursor(api.search, q=search).items(100)
    # tweets_data = [[tweet.user.name, tweet.text] for tweet in searched_tweet]
    # df = pd.DataFrame(tweets_data, columns=['user', 'tweet'])
    # df = df.drop_duplicates('tweet')

    # tweet_listner = TweetStreamListner(api)
    # stream = tweepy.Stream(api.auth, tweet_listner)
    # # stream.userstream('BJP','CONGRESS')
    # stream.filter(track=['NRC'], languages=['en'])
    print(api.me())


if __name__ == "__main__":
    main()
