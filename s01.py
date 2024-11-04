import tweepy
import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class WeightedCryptoSentimentIndicator:
    def __init__(self, twitter_credentials, reddit_credentials, update_interval='5min'):
        """
        Initialize the sentiment indicator with weighted metrics
        
        Args:
            twitter_credentials (dict): Twitter API credentials
            reddit_credentials (dict): Reddit API credentials
            update_interval (str): How often to update the indicator ('1min', '5min', '15min', '1h')
        """
        # API Setup
        auth = tweepy.OAuthHandler(
            twitter_credentials['api_key'],
            twitter_credentials['api_secret']
        )
        auth.set_access_token(
            twitter_credentials['access_token'],
            twitter_credentials['access_secret']
        )
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
        self.reddit = praw.Reddit(**reddit_credentials)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.update_interval = update_interval
        
        # Cache for storing data
        self.cache = {
            'last_update': None,
            'data': None
        }

    def calculate_tweet_weight(self, tweet):
        """Calculate weight based on engagement metrics"""
        likes_weight = np.log1p(tweet.favorite_count) * 1.0
        retweet_weight = np.log1p(tweet.retweet_count) * 1.5  # Higher weight for retweets
        reply_weight = np.log1p(tweet.reply_count) if hasattr(tweet, 'reply_count') else 0
        
        total_weight = 1 + likes_weight + retweet_weight + reply_weight
        return total_weight

    def calculate_reddit_weight(self, post):
        """Calculate weight based on Reddit engagement"""
        upvote_weight = np.log1p(post.score) * 1.0
        comment_weight = np.log1p(post.num_comments) * 1.5
        awards_weight = len(post.all_awardings) if hasattr(post, 'all_awardings') else 0
        
        total_weight = 1 + upvote_weight + comment_weight + awards_weight
        return total_weight

    def fetch_weighted_data(self, twitter_keyword, subreddit_name, lookback_hours=24):
        """Fetch and weight social media data"""
        now = datetime.utcnow()
        since = now - timedelta(hours=lookback_hours)
        
        # Fetch Twitter data with weights
        tweets = tweepy.Cursor(
            self.twitter_api.search_tweets,
            q=twitter_keyword,
            lang='en',
            tweet_mode='extended',
            since_id=since.timestamp()
        ).items(1000)

        tweet_data = []
        for tweet in tweets:
            weight = self.calculate_tweet_weight(tweet)
            sentiment = self.sentiment_analyzer.polarity_scores(tweet.full_text)['compound']
            
            tweet_data.append({
                'timestamp': tweet.created_at,
                'text': tweet.full_text,
                'sentiment': sentiment,
                'weight': weight,
                'weighted_sentiment': sentiment * weight,
                'source': 'twitter',
                'engagement': tweet.favorite_count + tweet.retweet_count
            })

        # Fetch Reddit data with weights
        subreddit = self.reddit.subreddit(subreddit_name)
        reddit_data = []
        
        for post in subreddit.new(limit=500):
            if datetime.utcfromtimestamp(post.created_utc) > since:
                weight = self.calculate_reddit_weight(post)
                sentiment = self.sentiment_analyzer.polarity_scores(post.title + ' ' + post.selftext)['compound']
                
                reddit_data.append({
                    'timestamp': datetime.utcfromtimestamp(post.created_utc),
                    'text': post.title + ' ' + post.selftext,
                    'sentiment': sentiment,
                    'weight': weight,
                    'weighted_sentiment': sentiment * weight,
                    'source': 'reddit',
                    'engagement': post.score + post.num_comments
                })

        return pd.DataFrame(tweet_data + reddit_data)

    def calculate_indicator(self, twitter_keyword, subreddit_name, timeframe='1h'):
        """Calculate the weighted sentiment indicator for specified timeframe"""
        # Check if we need to update the cache
        now = datetime.utcnow()
        if (self.cache['last_update'] is None or 
            (now - self.cache['last_update']).total_seconds() > pd.Timedelta(self.update_interval).total_seconds()):
            
            df = self.fetch_weighted_data(twitter_keyword, subreddit_name)
            self.cache['data'] = df
            self.cache['last_update'] = now
        else:
            df = self.cache['data']

        # Convert timestamps and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to desired timeframe and calculate weighted sentiment
        resampled = df.resample(timeframe).agg({
            'weighted_sentiment': 'sum',
            'weight': 'sum',
            'engagement': 'sum'
        })
        
        # Calculate final indicator value
        resampled['indicator'] = resampled['weighted_sentiment'] / resampled['weight']
        
        # Normalize to [-1, 1]
        resampled['indicator'] = 2 * (resampled['indicator'] - resampled['indicator'].min()) / \
                                (resampled['indicator'].max() - resampled['indicator'].min()) - 1
        
        return resampled

def main():
    st.title("Weighted Crypto Sentiment Indicator")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ['1min', '5min', '15min', '30min', '1h', '4h'],
        index=2
    )
    
    twitter_keyword = st.sidebar.text_input("Twitter Keyword", "#Bitcoin")
    subreddit = st.sidebar.text_input("Subreddit", "cryptocurrency")
    
    # Initialize indicator
    indicator = WeightedCryptoSentimentIndicator(
        twitter_credentials={'your_credentials_here'},
        reddit_credentials={'your_credentials_here'},
        update_interval='5min'
    )
    
    # Calculate and display indicator
    if st.button("Update Indicator"):
        data = indicator.calculate_indicator(twitter_keyword, subreddit, timeframe)
        
        # Plot indicator
        fig = px.line(data, y='indicator', 
                     title=f'Weighted Crypto Sentiment Indicator ({timeframe} timeframe)')
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Extremely Bullish")
        fig.add_hline(y=-0.7, line_dash="dash", line_color="green", annotation_text="Extremely Bearish")
        st.plotly_chart(fig)
        
        # Display current indicator value
        current_value = data['indicator'].iloc[-1]
        st.metric("Current Indicator Value", f"{current_value:.2f}")
        
        # Display engagement metrics
        st.metric("Total Engagement (Last Period)", 
                 int(data['engagement'].iloc[-1]))

if __name__ == "__main__":
    main()  
