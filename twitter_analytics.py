from __future__ import print_function
from __future__ import unicode_literals

from builtins import str, bytes, dict, int
from builtins import range

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.web import Twitter, hashtags
from pattern.db import Datasheet, pprint, pd


# This example retrieves tweets containing given keywords from Twitter.

try:
    # We'll store tweets in a Datasheet.
    # A Datasheet is a table of rows and columns that can be exported as a CSV-file.
    # In the first column, we'll store a unique id for each tweet.
    # We only want to add the latest tweets, i.e., those we haven't seen yet.
    # With an index on the first column we can quickly check if an id already exists.
    # The pd() function returns the parent directory of this script + any given path.
    table = Datasheet.load(pd("black.csv"))
    index = set(table.columns[0])
except:
    table = Datasheet()
    index = set()

engine = Twitter(language="en")

# With Twitter.search(cached=False), a "live" request is sent to Twitter:
# we get the most recent results instead of those in the local cache.
# Keeping a local cache can also be useful (e.g., while testing)
# because a query is instant when it is executed the second time.
prev = None
for i in range(2):
    print(i)
    for tweet in engine.search("#blacklivesmatter", start=prev, count=25, cached=False):
        print("")
        print(tweet.text)
        # print(tweet.author)
        print(tweet.date)
        print(hashtags(tweet.text))  # Keywords in tweets start with a "#".
        print("")
        # Only add the tweet to the table if it doesn't already exists.
        if len(table) == 0 or tweet.id not in index:
            table.append([tweet.id, tweet.text])
            index.add(tweet.id)
        # Continue mining older tweets in next iteration.
        prev = tweet.id

# Create a .csv in pattern/examples/01-web/
table.save(pd("black.csv"))

print("Total results: %s" % len(table))
print("")

# Print all the rows in the table.
# Since it is stored as a CSV-file it grows comfortably each time the script runs.
# We can also open the table later on: in other scripts, for further analysis, ...

pprint(table, truncate=100)

# Note: you can also search tweets by author:
# Twitter().search("from:tom_de_smedt")




# Import util libraries
import tweepy
import random
import numpy as np
import pandas as pd
import yaml
import warnings
import calendar
import time
from datetime import date
from PIL import Image
from collections import Counter

import re
import spacy.lang.es as es
import spacy.lang.en as en
from textblob import TextBlob
from wordcloud import WordCloud



# Import plot libraries
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Util function - Read dict from yaml file
def get_dict_from_yaml(yaml_path):
    result = dict()
    
    with open(yaml_path) as f:
        yaml_file = f.read()
        result = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    return result

# Util function - Plot column chart
def plot_col_chart(df, figsize, x_var, y_var, title, color='green', legend=None, x_label=None):
    fig, ax = plt.subplots()
    df.plot.bar(ax=ax, x=x_var, y=y_var, color=color, alpha=0.75, figsize=figsize)
    if legend:
        ax.legend(legend)
    else:
        ax.get_legend().remove()
    if x_label:
        x = np.arange(len(x_label))
        plt.xticks(x, x_label, rotation=45)
    else:
        plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.xlabel(x_var.capitalize())
    plt.ylabel(y_var.capitalize())
    plt.show()

# Util function - Plot column chart
def plot_bar_chart(df, figsize, x_var, y_var, title, color='blue', legend=None):
    fig, ax = plt.subplots()
    df.plot.barh(ax=ax, x=x_var, y=y_var, alpha=0.75, figsize=figsize)
    if legend:
        ax.legend(legend)
    else:
        ax.get_legend().remove()
    plt.title(title, fontsize=16)
    plt.xlabel(y_var.capitalize())
    plt.ylabel(x_var.capitalize())
    plt.show()


# Read twitter bot credentials
yaml_path = 'config/credentials_example.yml'
twt_login = get_dict_from_yaml(yaml_path)

# Setup bot credentials
consumer_key = twt_login['consumer_key']
consumer_secret = twt_login['consumer_secret']
access_token = twt_login['access_token']
access_token_secret = twt_login['access_token_secret']

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

# Util function - Fetch tweets list from a specific user
# Note: Twitter only allows access to a users most recent 3240 tweets with this method
def get_all_tweets(api, screen_name):
    all_tweets = []
    
    # Make initial request for most recent tweets (200 is the maximum allowed count)
    try:
        new_tweets = api.user_timeline(screen_name = screen_name, count=200, tweet_mode='extended')

        # Save most recent tweets
        all_tweets.extend(new_tweets)

        # Save the id of the oldest tweet less one
        oldest = all_tweets[-1].id - 1

        # Keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:

            # All subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name = screen_name, count=200, tweet_mode='extended', max_id=oldest)

            # Save most recent tweets
            all_tweets.extend(new_tweets)

            # Update the id of the oldest tweet less one
            oldest = all_tweets[-1].id - 1
    
    except (socket.timeout, exceptions.ReadTimeoutError, exceptions.ProtocolError, tweepy.TweepError) as e:
        print('Error:', e)
    
    # Transform the tweepy tweets into an array that contains the relevant fields of each tweet
    tweet_list = []
    for tweet in all_tweets:
        new_tweet = {
            'id': tweet.id_str,
            'created_at': tweet.created_at,
            'message': tweet.full_text,
            'lang': tweet.lang,
            'hashtags': [ht['text'] for ht in tweet.entities['hashtags']],
            'retweet_count': tweet.retweet_count,
            'favorite_count': tweet.favorite_count,
            'retweeted': tweet.retweeted,
            'source': tweet.source
        }
        tweet_list.append(new_tweet)
    
    return tweet_list


# twitter api key -- DeBOwkh5gtglRgom2Rxjr9IfQ
# API secret key  ---  KaQ71tpJQILsxvk3VTYpu5RuI8iBTGRT1WKoFLmDlj9dhpVK96
# Bearer token ---   AAAAAAAAAAAAAAAAAAAAAIf%2FJAEAAAAAE0kdMn2FMVaVnvCgOj1sCqTlqug%3DEjQpKRGBkIvi1b2Rycx6347coFEu0JbC4GEYHUolYbDLZ99r6i
# Accesstoken---  1321049834405597184-gjCTI4edyuGZcnV3Kr2gL2xBgGAkX8
# Accesstokensecret---  SbPEXB2fLRQHBL2r7h7NqqzhrRKLJFtagMMq2p6J4Ke2R

# Show user account details
tw_user_name = "@SiddharthPokh16"
user = api.get_user(screen_name=tw_user_name)

print("User details:")
print(user.name)
print(user.description)
print(user.location)
print(user.created_at)


# Fetching tweet list from a specific user
tweet_list = get_all_tweets(api, screen_name=tw_user_name)

# Show the last tweet
print(tweet_list)

# Get a list of all followers of a twitter account
followers = []
for page in tweepy.Cursor(api.followers, screen_name=tw_user_name, wait_on_rate_limit=True, count=200).pages():
    try:
        followers.extend(page)
    except tweepy.TweepError as e:
        time.sleep(10)

n_followers = len(followers)
print('Number of followers: %s' % n_followers)

# Get year the account was created
flw_creation = Counter()
for flw in followers:
    created_at = flw.created_at.date()
    year = created_at.year
    flw_creation[year] += 1

# Cooking dataframe
df = pd.DataFrame.from_records(flw_creation.most_common(), columns = ['year', 'frequency']).sort_values(by=['year'])

# Plot creation of followers account
figsize = (12, 7)
x_var = 'year'
y_var = 'frequency'
title = 'Creation of followers account'
color = 'indigo'
plot_col_chart(df, figsize, x_var, y_var, title, color)

# Get year the account was created
flw_following_count = []
flw_followers_count = []

for flw in followers:
    flw_following_count.append(flw.friends_count)
    flw_followers_count.append(flw.followers_count)

# Showing followers stats
print('Average accounts followed by followers: %.2f' % np.mean(flw_following_count))
print('Median accounts followed by followers: %.2f' % np.median(flw_following_count))
print('Average of followers of my followers: %.2f' % np.mean(flw_followers_count))
print('Median of followers of my followers: %.2f' % np.median(flw_followers_count))

# Tweet length
n_tweets = len(tweet_list)
tweets_len = []

for tweet in tweet_list:
    tweets_len.append(len(tweet['message']))

# Showing stats
print('Number of tweets: %s' % n_tweets)
print('Minimum length: %s' % np.min(tweets_len))
print('Maximum length: %s' % np.max(tweets_len))
print('Average length: %.2f' % np.mean(tweets_len))
print('Standard deviation: %.2f' % np.std(tweets_len))

# Plot histograma of the tweet length
fig = plt.figure(figsize = (12, 7))
plt.hist(tweets_len, 20, facecolor='green', alpha=0.75)
plt.title('Tweet Length Histogram', fontsize=16)
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Get retweet and favorite stats
retweet_count = []
favorite_count = []

for tweet in tweet_list:
    if not tweet['retweeted']:
        retweet_count.append(tweet['retweet_count'])
        favorite_count.append(tweet['favorite_count'])

# Showing statistics of account tweets
print('Retweets total: %s' % np.sum(retweet_count))
print('Retweets average: %.2f' % np.mean(retweet_count))
print('Favorites total: %s' % np.sum(favorite_count))
print('Favorites average: %.2f' % np.mean(favorite_count))

# Show top 10 tweets 
retweet_2mean = np.mean(retweet_count) + np.std(retweet_count)
favorite_2mean = np.mean(favorite_count) + np.std(favorite_count)
print('Limits >> retweet_2mean: %.2f and favorite_2mean: %.2f' % (retweet_2mean, favorite_2mean))

df = pd.DataFrame(columns=['Created At', 'Retweet Count', 'Favorite Count', 'Hashtags', 'Language', 'Message'])
ix = 0

# Filter tweets
for tw in tweet_list:
    if tw['retweet_count'] > retweet_2mean and tw['favorite_count'] > favorite_2mean:
        data_row = (tw['created_at'], tw['retweet_count'], tw['favorite_count'], tw['hashtags'], tw['lang'], tw['message'])
        df.loc[ix] = data_row
        ix += 1

# Show dataframe
df.sort_values(by=['Retweet Count', 'Favorite Count'], ascending=False)

# Read iso639_2 languages codes
filename = 'config/iso_lang_codes.csv'
df = pd.read_csv(filename, names = ['iso_code', 'language'], skiprows=1)
lang_dict = dict(zip(list(df.iso_code), list(df.language)))
print('Languages dict length:', len(lang_dict))

# Count the language of the tweets
lang_list = Counter()

for tweet in tweet_list:
    tweet_lang = tweet['lang']
    if tweet_lang in lang_dict.keys():
        lang = lang_dict[tweet['lang']]
        lang_list[lang] += 1
    else:
        print('Missing:', tweet_lang)

print('Total tweets language: %s' % len(lang_list))

# Cooking dataframe
df = pd.DataFrame.from_records(lang_list.most_common(), columns = ['language', 'frequency'])

# Plot the language of the tweets
x_var = 'language'
y_var = 'frequency'
title = 'Tweets by Languages'
figsize = (7, 7)
plot_col_chart(df, figsize, x_var, y_var, title)

# Count the language of the tweets
tweets_year = Counter()
tweets_day = Counter()

for tweet in tweet_list:
    created_at = tweet['created_at'].date()
    weekday = created_at.weekday()
    year = created_at.year
    
    tweets_day[weekday] += 1
    tweets_year[year] += 1

# Cooking dataframe
df = pd.DataFrame.from_records(list(tweets_day.items()), columns = ['weekday', 'frequency']).sort_values(by=['weekday'])
x = np.arange(7)
x_label = [calendar.day_name[d] for d in x]

# Plot tweets by weekday
x_var = 'weekday'
y_var = 'frequency'
title = 'Tweets by Weekday'
figsize = (12, 7)
color = 'green'
x_label = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plot_col_chart(df, figsize, x_var, y_var, title, color, None, x_label)

df = pd.DataFrame.from_records(tweets_year.most_common(), columns = ['year', 'frequency']).sort_values(by=['year'])

# Plot tweets by year
x_var = 'year'
y_var = 'frequency'
title = 'Tweets by Year'
figsize = (12, 7)
plot_col_chart(df, figsize, x_var, y_var, title)

# Count the source of the tweets
tweets_source = Counter()

for tweet in tweet_list:
    year = tweet['source']
    tweets_source[year] += 1

print('Total tweeting sources: %s' % len(tweets_source))

# Cooking dataframe
df = pd.DataFrame.from_records(tweets_source.most_common(), columns = ['source', 'frequency'])

# Plot tweeting Sources
x_var = 'source'
y_var = 'frequency'
title = 'Tweets by Sources'
figsize = (12, 7)
plot_col_chart(df, figsize, x_var, y_var, title)


# Util function - Clean tweet text

def dq_clean_text(text):
    clean_text = text.lower()
    clean_text = re.sub(rx, ' ', clean_text)
    clean_text = re.sub(r'\.+', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


# Global NLP variables
n_top = 20
punt_marks = ['\n', '\'', '\\', '/', '¡', '!', '¿', '?', '.', ',', ';', ':', '#', '$', '%', '&', '(', ')', '"']
rx = '[' + re.escape(''.join(punt_marks)) + ']'

# Counter of used words
word_list = Counter()

# Create list of words
for tweet in tweet_list:
    tweet_text = tweet['message']
    
    # Clean tweet text
    clean_text = dq_clean_text(tweet_text)
    
    # Save mentioned accounts
    for word in clean_text.split(' '):
        if len(word) and not word.isnumeric():
            word_list[word] += 1

print('Number of words used: %s' % len(word_list))
print('Average words per tweet: %.2f' % (len(word_list) / n_tweets))


# Most mentioned accounts list
tw_accounts = Counter()

# Save mentioned accounts
for word, freq in list(word_list.items()):
    if word[0] == '@' and word != tw_user_name:
        tw_accounts[word] = freq

print('Total different mentioned accounts: %s' % len(tw_accounts))
print('Average mentioned accounts per tweet: %.2f' % (sum(tw_accounts.values())/n_tweets))


# Cooking dataframe
top_accounts = tw_accounts.most_common(n_top)
df = pd.DataFrame.from_records(top_accounts, columns = ['account', 'frequency']).sort_values(by=['frequency'])

# Plot top N mentioned accounts
x_var = 'account'
y_var = 'frequency'
title = 'Top %s Mentioned Twitter Accounts' % n_top
figsize = (8, 10)
plot_bar_chart(df, figsize, x_var, y_var, title)


# Count the hashtags in the tweets and return the N most repeated 
hashtags = Counter()

for t in tweet_list:
    for ht in t['hashtags']:
        ht = '#'+ht.lower()
        hashtags[ht] += 1

print('Total used different hashtags: %s' % len(hashtags))
print('Average hashtags per tweet: %.2f' % (sum(hashtags.values())/n_tweets))


# Cooking dataframe
top_hashtags = hashtags.most_common(n_top)
df = pd.DataFrame.from_records(top_hashtags, columns = ['hashtag', 'frequency']).sort_values(by=['frequency'])


# Plot the most common hashtags in tweets
x_var = 'hashtag'
y_var = 'frequency'
title = '%s Most used Hashtags' % n_top
figsize = (8, 10)
plot_bar_chart(df, figsize, x_var, y_var, title)

# Add word-freq to Dataframe
df = pd.DataFrame(columns=['word', 'frequency'])
ix = 0

# Filter twitter accounts
for word, freq in word_list.most_common():
    if ix == n_top:
        break
    elif len(word) > 2 and word[0] != '@':
        df.loc[ix] = (word, freq)
        ix += 1

# Sorting data by frequency
df = df.sort_values(by=['frequency'])

# Plot the most common words in tweets
x_var = 'word'
y_var = 'frequency'
title = '%s Most used Words' % n_top
figsize = (8, 10)
plot_bar_chart(df, figsize, x_var, y_var, title)


# Get Spanish stopwords
stopwords_es = es.stop_words.STOP_WORDS
stopwords_en = en.stop_words.STOP_WORDS | set({'http', 'https'})

# Add word-freq to Dataframe
df = pd.DataFrame(columns=['word', 'frequency'])
ix = 0

# Filter English and Spanihs stopwords
for word, freq in word_list.most_common():
    if ix == n_top:
        break
    elif (word not in stopwords_es) and (word not in stopwords_en) and (len(word) > 2) and (word[0] != '@'):
        df.loc[ix] = (word, freq)
        ix += 1

# Sorting data by frequency
df = df.sort_values(by=['frequency'])

# Plot the most common words in tweets
x_var = 'word'
y_var = 'frequency'
title = '%s Most used Words w/o Stopwords' % n_top
figsize = (8, 10)
plot_bar_chart(df, figsize, x_var, y_var, title)


# Calculate most common bigrams and reconstruct full text with used words
new_clean_text = user.description
bigram_list = Counter()

# Create list of words
for tweet in tweet_list:
    tweet_text = tweet['message']
    clean_text = dq_clean_text(tweet_text)
    
    tokens = clean_text.split(' ')
    bigram = ''
    last_word = tokens[0]
    
    for i in range(1, len(tokens)):
        word = tokens[i]
        if (word not in stopwords_es) and (word not in stopwords_en) and (len(word) > 2) and (word[0] != '@'):
            
            # Reconstructing the clean text (without stop-words)
            new_clean_text += ' ' + word
            
            # Add bigrams-freq to Dataframe
            bigram = last_word + '-' + word
            last_word = word
            bigram_list[bigram] += 1

print('Total different bigrams: %s' % len(bigram_list))

# Cooking dataframe
top_bigrams = bigram_list.most_common(n_top)
df = pd.DataFrame.from_records(top_bigrams, columns = ['bigram', 'frequency']).sort_values(by=['frequency'])


# Plot the most common words in tweets
x_var = 'bigram'
y_var = 'frequency'
title = '%s Most used Bigrams' % n_top
figsize = (8, 10)
plot_bar_chart(df, figsize, x_var, y_var, title)

# Import mask
# mask = np.array(Image.open('../img/twitter-mask.png'))

# Create a Word-cloud
# wc = WordCloud(background_color="white", max_words=250, mask=mask, collocations=False, margin=2)
# wc = wc.generate(new_clean_text)

# # Plot a Word-cloud of words of tweets
# plt.figure(figsize = (16, 10))
# image = plt.imshow(wc, interpolation = "bilinear")
# plt.title('%s WordCloud\n' % tw_user_name, fontsize=16)
# plt.axis("off")

# # Save the Word-cloud image
# plt.savefig('../img/wordcloud.png')
# plt.show()

# Functions to discretize the sentiment analysis results
def get_polarity(polarity):
    if polarity >= 0.20:
        return 'positive'
    elif polarity <= -0.20:
        return 'negative'
    return 'neutral'

def get_subjectivity(subjectivity):
    if subjectivity >= 0.5:
        return 'subjective'
    return 'objective'


# Calculate the sentiment of the tweet
allow_translation = True
lang_dest = 'en'
polarity_list = Counter({'positive': 0, 'negative': 0, 'neutral': 0})
subjectivity_list = Counter({'subjective': 0, 'objective': 0})

for tweet in tweet_list:
    message = tweet['message']
    lang_source = tweet['lang']
    
    # If the language of the tweet is different from English, it is translated
    if lang_source == lang_dest or allow_translation:
        analysis = TextBlob(message)
        try:
            if lang_source != lang_dest:
                analysis = analysis.translate(to=lang_dest)
                time.sleep(0.1)
            
            # Apply sent-analysis
            polarity = get_polarity(analysis.sentiment.polarity)
            subjectivity = get_subjectivity(analysis.sentiment.subjectivity)

            # Save results
            polarity_list[polarity] += 1
            subjectivity_list[subjectivity] += 1
        except:
            pass


# Cooking dataframe
df = pd.DataFrame.from_records(list(polarity_list.items()), columns = ['polarity', 'frequency'])
df['perc'] = (100 * df['frequency'] / sum(df['frequency']))
print("df----:" % df)


# Plot the polarity of tweets
x_var = 'polarity'
y_var = 'frequency'
title = 'Polarity of Tweets'
figsize = (7, 7)
plot_col_chart(df, figsize, x_var, y_var, title)


# Cooking dataframe
df = pd.DataFrame.from_records(list(subjectivity_list.items()), columns = ['subjectivity', 'frequency'])
df['perc'] = (100 * df['frequency'] / sum(df['frequency']))
df

# Plot the subjectivity of tweets
x_var = 'subjectivity'
y_var = 'frequency'
title = 'Subjectivity of Tweets'
figsize = (7, 7)
plot_col_chart(df, figsize, x_var, y_var, title)




