from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
import math
import time
import re
import numpy as np
import logging
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

app = Flask(__name__)
CORS(app) 
analyzer = SentimentIntensityAnalyzer()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

reddit = praw.Reddit(
    client_id='',  
    client_secret='',
    user_agent='sentiscope_app'
)

class TwitterScraper:
    def __init__(self, headless=True, wait_time=10):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        
        self.driver = None
        self.base_url = "https://nitter.net/search?f=tweets&q="
        self.wait_time = wait_time
        
    def start_driver(self):
        self.driver = webdriver.Firefox(options=self.options)
        self.driver.set_page_load_timeout(30)
        
    def close_driver(self):
        """Close the driver."""
        if self.driver:
            self.driver.quit()
    
    def format_timestamp(self, timestamp_text):
        try:
            if "·" in timestamp_text:
                parts = timestamp_text.split("·")
                date_part = parts[0].strip()
                time_part = parts[1].strip().replace(" UTC", "")
                
                combined = f"{date_part} {time_part}"
                return datetime.strptime(combined, "%b %d, %Y %I:%M %p")
            # Handle relative timestamps
            elif "ago" in timestamp_text:
                now = datetime.now()
                if "minute" in timestamp_text or "min" in timestamp_text:
                    match = re.search(r'(\d+)', timestamp_text)
                    if match:
                        minutes = int(match.group(1))
                        return now - timedelta(minutes=minutes)
                elif "hour" in timestamp_text or "hr" in timestamp_text:
                    match = re.search(r'(\d+)', timestamp_text)
                    if match:
                        hours = int(match.group(1))
                        return now - timedelta(hours=hours)
                elif "day" in timestamp_text:
                    match = re.search(r'(\d+)', timestamp_text)
                    if match:
                        days = int(match.group(1))
                        return now - timedelta(days=days)
                else:
                    return now
            else:
                return datetime.strptime(timestamp_text, "%d %b %Y, %H:%M:%S")
        except Exception as e:
            logger.warning(f"Could not parse timestamp: {timestamp_text}. Error: {e}")
            return datetime.now()  # Default to current time if parsing fails
            
    def process_tweet(self, tweet):
        try:
            try:
                tweet_text_element = tweet.find_element(By.CSS_SELECTOR, ".tweet-content")
                tweet_text = tweet_text_element.text
            except NoSuchElementException:
                logger.warning("Tweet content not found, skipping")
                return None, None, None
            
            try:
                timestamp_element = tweet.find_element(By.CSS_SELECTOR, ".tweet-date a")
                timestamp_text = timestamp_element.get_attribute("title")
                if not timestamp_text:
                    timestamp_text = timestamp_element.text
                
                tweet_datetime = self.format_timestamp(timestamp_text)
                tweet_timestamp = tweet_datetime.strftime("%H:%M:%S")  # Format as HH:MM:SS for frontend
            except NoSuchElementException:
                logger.warning("Tweet timestamp not found, using current time")
                tweet_timestamp = datetime.now().strftime("%H:%M:%S")
            
            engagement = {}
            try:
                stats_element = tweet.find_element(By.CSS_SELECTOR, ".tweet-stats")
                stat_elements = stats_element.find_elements(By.CSS_SELECTOR, ".icon-container")
                
                if len(stat_elements) >= 1:
                    engagement['replies'] = self._parse_count(stat_elements[0].text)
                if len(stat_elements) >= 2:
                    engagement['retweets'] = self._parse_count(stat_elements[1].text)
                if len(stat_elements) >= 3:
                    engagement['quotes'] = self._parse_count(stat_elements[2].text)
                if len(stat_elements) >= 4:
                    engagement['likes'] = self._parse_count(stat_elements[3].text)
                
            except (NoSuchElementException, IndexError):
                engagement = {'likes': 1, 'retweets': 0, 'replies': 0, 'quotes': 0}
                
            return tweet_text, tweet_timestamp, engagement
            
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return None, None, None
    
    def _parse_count(self, count_text):
        if not count_text or count_text.strip() == '':
            return 0
            
        count_text = count_text.strip().lower()
        if 'k' in count_text:
            return int(float(count_text.replace('k', '')) * 1000)
        elif 'm' in count_text:
            return int(float(count_text.replace('m', '')) * 1000000)
        else:
            try:
                return int(count_text)
            except ValueError:
                return 0
    
    def calculate_weighted_score(self, text, engagement):
        sentiment = analyzer.polarity_scores(text)['compound']
        
        sentiment = sentiment * 0.7  # Dampen the sentiment impact
        
        likes_weight = math.sqrt(engagement.get('likes', 1) + 1) * 0.2
        retweets_weight = math.sqrt(engagement.get('retweets', 0) + 1) * 0.3
        replies_weight = math.sqrt(engagement.get('replies', 0) + 1) * 0.2
        
        engagement_weight = min(likes_weight + retweets_weight + replies_weight, 3.0)
        
        weighted_score = sentiment * engagement_weight
        
        return max(min(weighted_score, 0.8), -0.8)
    
    def get_sentiment(self, query, limit=250, max_pages=25):
        if not self.driver:
            self.start_driver()
            
        scores = []
        timestamps = []
        search_url = f"{self.base_url}{query}"
        current_page = 1
        
        try:
            logger.info(f"Starting to scrape up to {limit} tweets for query: {query}")
            
            self.driver.get(search_url)
            logger.info(f"Loaded initial page: {search_url}")
            
            wait = WebDriverWait(self.driver, self.wait_time)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".timeline-item")))
            
            while len(scores) < limit and current_page <= max_pages:
                logger.info(f"Processing page {current_page}...")
                
                tweets = self.driver.find_elements(By.CSS_SELECTOR, ".timeline-item")
                
                valid_tweets = []
                for t in tweets:
                    try:
                        class_attr = t.get_attribute("class")
                        if class_attr and "timeline-item" in class_attr:
                            valid_tweets.append(t)
                    except:
                        continue
                
                logger.info(f"Found {len(valid_tweets)} tweets on page {current_page}")
                
                for tweet in valid_tweets:
                    if len(scores) >= limit:
                        break
                        
                    tweet_text, tweet_timestamp, engagement = self.process_tweet(tweet)
                    
                    if tweet_text and tweet_timestamp and engagement:
                        # Calculate sentiment with weighting
                        weighted_score = self.calculate_weighted_score(tweet_text, engagement)
                        
                        scores.append(weighted_score)
                        timestamps.append(tweet_timestamp)
                        
                        logger.info(f"Tweet {len(scores)}/{limit}: Score={weighted_score:.2f}, Time={tweet_timestamp}")
                
                if len(scores) >= limit:
                    logger.info(f"Reached the requested count of {limit} tweets")
                    break
                
                try:
                    load_more_elements = self.driver.find_elements(By.CSS_SELECTOR, ".show-more a")
                    
                    load_more_link = None
                    for element in load_more_elements:
                        if element.text == "Load more":
                            load_more_link = element
                            break
                    
                    if not load_more_link:
                        logger.warning("No 'Load more' button found, ending pagination")
                        break
                    
                    load_more_link.click()
                    current_page += 1
                    
                    # Wait for the new page to load
                    time.sleep(3)
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".timeline-item")))
                    
                except Exception as e:
                    logger.error(f"Error navigating to next page: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
        
        return scores, timestamps

def get_twitter_sentiment(query, limit=250):
    scraper = TwitterScraper(headless=True)
    try:
        scores, timestamps = scraper.get_sentiment(query, limit=limit)
        logger.info(f"Twitter results for '{query}': {len(scores)} scores, {len(timestamps)} timestamps")
        return scores, timestamps
    except Exception as e:
        logger.error(f"Twitter scraping error: {e}")
        return [], []
    finally:
        scraper.close_driver()

def calculate_reddit_score(text, upvotes, comments, awards):
    sentiment = analyzer.polarity_scores(text)['compound']
    
    sentiment = sentiment * 0.7
    
    # Use square root for more moderate scaling of engagement
    upvote_weight = math.sqrt(upvotes + 1) * 0.2
    comment_weight = math.sqrt(comments + 1) * 0.3
    award_weight = math.sqrt(awards + 1) * 0.2
    
    engagement_weight = min(upvote_weight + comment_weight + award_weight, 3.0)
    
    weighted_score = sentiment * engagement_weight
    
    return max(min(weighted_score, 0.8), -0.8)

def get_reddit_sentiment(query, limit=250):
    scores = []
    timestamps = []

    try:
        logger.info(f"Starting Reddit search for '{query}' with limit={limit}")
        
        for i, post in enumerate(reddit.subreddit("all").search(query, limit=limit)):
            if i >= limit:
                break

            text = post.title + " " + (post.selftext or '')
            
            upvotes = post.score if hasattr(post, 'score') else 0
            comments = post.num_comments if hasattr(post, 'num_comments') else 0
            awards = len(post.all_awardings) if hasattr(post, 'all_awardings') else 0
            
            score = calculate_reddit_score(text, upvotes, comments, awards)
            scores.append(score)

            if hasattr(post, 'created_utc'):
                timestamp = datetime.fromtimestamp(post.created_utc).strftime("%H:%M:%S")
            else:
                timestamp = (datetime.now() - timedelta(minutes=i + limit)).strftime("%H:%M:%S")
                
            timestamps.append(timestamp)
            
            logger.info(f"Reddit post {i+1}/{limit}: Score={score:.2f}, Upvotes={upvotes}, " +
                       f"Comments={comments}, Time={timestamp}")

    except Exception as e:
        logger.error(f"Reddit error: {e}")
        return [], []
    
    logger.info(f"Reddit sentiment results for '{query}': {len(scores)} data points")
    return scores, timestamps

def advanced_smoothing(scores, timestamps, window_size=10):
    
        
    if len(scores) <= 1:
        return scores, timestamps
    
    np_scores = np.array(scores)
    
    def remove_outliers(data, threshold=1.5):
        if len(data) < 5:  # Need at least 5 points for meaningful outlier detection
            return data
            
        # Calculate rolling median
        window = min(5, len(data) // 2)
        if window < 2:
            window = 2
            
        medians = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            medians[i] = np.median(data[start:end])
        
        # Calculate deviation from median
        deviation = np.abs(data - medians)
        
        # Get median of deviations
        mad = np.median(deviation)
        
        # Replace outliers with local median
        if mad > 0:
            modified = data.copy()
            outlier_indices = np.where(deviation > threshold * mad)[0]
            modified[outlier_indices] = medians[outlier_indices]
            return modified
        return data
    
    np_scores = remove_outliers(np_scores)
    
    def exp_moving_average(data, alpha=0.3):
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            
        return smoothed
    
    np_scores = exp_moving_average(np_scores)
    
    try:
        if len(np_scores) >= 5:
            sigma = max(1, min(window_size // 4, 3))  # Adaptive sigma
            np_scores = gaussian_filter1d(np_scores, sigma=sigma)
    except:
        logger.warning("Gaussian smoothing failed, falling back to simple smoothing")
        # Apply simple moving average as fallback
        temp = np.zeros_like(np_scores)
        for i in range(len(np_scores)):
            start = max(0, i - window_size // 2)
            end = min(len(np_scores), i + window_size // 2 + 1)
            temp[i] = np.mean(np_scores[start:end])
        np_scores = temp
    
    for i in range(len(np_scores)):
        if -0.1 < np_scores[i] < 0.1:
            # Push slightly away from zero to avoid flat lines
            direction = 1 if np_scores[i] >= 0 else -1
            np_scores[i] = direction * max(0.1, abs(np_scores[i]))
    
    smoothed_scores = np_scores.tolist()
    
    smoothed_scores = [max(min(s, 0.8), -0.8) for s in smoothed_scores]
    
    logger.info(f"Applied advanced smoothing: {len(scores)} points → {len(smoothed_scores)} smoothed points")
    
    return smoothed_scores, timestamps

def generate_mock_data(keyword):
    scores = []
    timestamps = []
    now = datetime.now()
    
    seed = sum(ord(c) for c in keyword)
    np.random.seed(seed)
    
    x = np.linspace(0, 4*np.pi, 50)
    base_trend = 0.3 * np.sin(x) + 0.15 * np.sin(2.5*x + 1.5) + 0.05 * np.sin(5*x + 0.6)
    
    noise = np.random.normal(0, 0.05, len(base_trend))
    smooth_trend = base_trend + noise
    
    keyword_bias = (len(keyword) % 10) / 20.0 - 0.25
    smooth_trend += keyword_bias
    
    smooth_trend = np.clip(smooth_trend, -0.8, 0.8)
    
    segment_start = np.random.randint(0, len(smooth_trend) - 30)
    selected_trend = smooth_trend[segment_start:segment_start+30]
    
    for i in range(len(selected_trend)):
        scores.append(float(selected_trend[i]))
        mock_time = now - timedelta(minutes=len(selected_trend)-i)
        timestamps.append(mock_time.strftime("%H:%M:%S"))
    
    return scores, timestamps

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    keyword = data.get('keyword', '')
    
    if not keyword:
        return jsonify({'error': 'Keyword is required'}), 400
    
    logger.info(f"Starting sentiment analysis for keyword: '{keyword}'")
    
    twitter_limit = 250
    twitter_scores, twitter_timestamps = get_twitter_sentiment(keyword, limit=twitter_limit)
    logger.info(f"Twitter analysis complete: {len(twitter_scores)} data points")
    
    reddit_limit = 250
    reddit_scores, reddit_timestamps = get_reddit_sentiment(keyword, limit=reddit_limit)
    logger.info(f"Reddit analysis complete: {len(reddit_scores)} data points")
    
    all_scores = twitter_scores + reddit_scores
    all_timestamps = twitter_timestamps + reddit_timestamps
    
    if len(all_scores) < 5:
        logger.info(f"Not enough data, generating mock data for '{keyword}'")
        mock_scores, mock_timestamps = generate_mock_data(keyword)
        all_scores = mock_scores
        all_timestamps = mock_timestamps
    else:
        timestamp_score_pairs = sorted(zip(all_timestamps, all_scores), key=lambda x: x[0])
        all_timestamps = [pair[0] for pair in timestamp_score_pairs]
        all_scores = [pair[1] for pair in timestamp_score_pairs]
    
    smoothed_scores, final_timestamps = advanced_smoothing(all_scores, all_timestamps, window_size=15)
    
    if len(smoothed_scores) < 50 and len(smoothed_scores) > 5:
        expanded_scores = []
        expanded_timestamps = []
        
        for i in range(len(smoothed_scores) - 1):
            expanded_scores.append(smoothed_scores[i])
            expanded_timestamps.append(final_timestamps[i])
            
            # Add interpolated points
            num_points = max(1, int(50 / len(smoothed_scores)) - 1)
            for j in range(num_points):
                ratio = (j + 1) / (num_points + 1)
                interp_score = smoothed_scores[i] * (1 - ratio) + smoothed_scores[i+1] * ratio
                expanded_scores.append(interp_score)
                
                try:
                    time1 = datetime.strptime(final_timestamps[i], "%H:%M:%S")
                    time2 = datetime.strptime(final_timestamps[i+1], "%H:%M:%S")
                    diff_seconds = (time2 - time1).total_seconds()
                    interp_time = time1 + timedelta(seconds=diff_seconds * ratio)
                    expanded_timestamps.append(interp_time.strftime("%H:%M:%S"))
                except:
                    expanded_timestamps.append(final_timestamps[i])
        
        expanded_scores.append(smoothed_scores[-1])
        expanded_timestamps.append(final_timestamps[-1])
        
        smoothed_scores = expanded_scores
        final_timestamps = expanded_timestamps
    
    if smoothed_scores:
        weights = np.linspace(0.5, 1.0, min(10, len(smoothed_scores)))
        recent_scores = smoothed_scores[-len(weights):]
        current_score = np.average(recent_scores, weights=weights)
    else:
        current_score = 0
    
    if len(smoothed_scores) > 100:
        step = len(smoothed_scores) // 100 + 1
        final_scores = smoothed_scores[::step][:100]
        final_timestamps = final_timestamps[::step][:100]
    else:
        final_scores = smoothed_scores
        final_timestamps = final_timestamps
    
    logger.info(f"Analysis complete for '{keyword}': {len(final_scores)} smoothed data points, " 
                f"Current score: {current_score:.4f}")
    
    response = {
        'current_score': current_score,
        'historical_scores': final_scores,  # Smoothed scores for chart
        'timestamps': final_timestamps,     # Corresponding timestamps
    }
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

if __name__ == '__main__':
    logger.info("Starting Enhanced Sentiment Analysis API")
    app.run(host='0.0.0.0', port=5000, debug=True)