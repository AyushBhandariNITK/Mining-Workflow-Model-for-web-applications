
from __future__ import unicode_literals
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import nltk
import plotly.express as px


nltk.download('vader_lexicon')
matplotlib.style.use('ggplot')

Amazon_Meta_Data = pd.read_csv("config/Amazon_Unlocked_Mobile.csv")


Amazon_Meta_Data.head(2)
print(Amazon_Meta_Data.columns)
print(Amazon_Meta_Data.dtypes)


Reviews = Amazon_Meta_Data['Reviews']
print("\nlength of review:- ")
print(len(Reviews))

Brand_Name = Amazon_Meta_Data['Brand Name'].str.upper()
print("\nBrand name:- ")
# print(Brand_Name)
print(end="\n")
print("Brand Name\tcount")
print(Brand_Name.value_counts().head(10))



df=DataFrame(Amazon_Meta_Data.head(200),columns=['Brand Name','Price'])
df.sort_values(["Price","Brand Name"],axis=0,ascending=True,inplace=True)

# df=px.data.tips()
# df["Brand Name"]=df["Brand Name"].astype(str)

fig=px.bar(df,x="Price",y="Brand Name",orientation="h")
fig.show()

Price = Amazon_Meta_Data['Price']
print("price mean...." , Price.mean())
print("price median..." , Price.median())

# %time
table = pd.pivot_table(Amazon_Meta_Data,
            values = ['Price'],
            index = ['Brand Name'], 
                       columns= [],
                       aggfunc=[np.mean, np.median], 
                       margins=True)

#table
# Amazon_Meta_Data1=Amazon_Meta_Data.set_index(['Brand Name','Price']).sort_index()

# data=Amazon_Meta_Data1.loc['Brand Name','Price']
# pp.plot(data.index, data.values)


Customer_Ratings = Amazon_Meta_Data.groupby(
    'Brand Name'
    ).Rating.agg(
        ['count', 'min', 'max']
    ).sort_values(
        'count', ascending=False
    )

print(Customer_Ratings.head(15))

Product_Ratings = Amazon_Meta_Data.groupby(
    'Product Name'
    ).Rating.agg(
        ['count', 'min', 'max']
    ).sort_values(
        'count', ascending=False
    )
print(Product_Ratings.head(15))

pivot = pd.pivot_table(Amazon_Meta_Data,
            values = ['Rating'],
            index =  ['Brand Name'],
                       columns= [],
                       aggfunc=[np.sum, np.mean, np.count_nonzero, np.std], 
                       margins=True, fill_value=0).sort_values(by=('count_nonzero', 'Rating'), ascending=False).fillna('')
top_10_brands = pivot.reindex().head(n=11)
print(top_10_brands)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sample_review = Reviews[:5]
sentiment = SentimentIntensityAnalyzer()

for sentences in sample_review:
    print(sentences)
    ss = sentiment.polarity_scores(sentences)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]))
    print(sentences)

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print(len(Reviews))

Test = Reviews[:100000]
type(Test)
from gensim.models import Word2Vec
from gensim.models import word2vec

Test.to_csv("config/w2v.txt")

data = word2vec.Text8Corpus("config/w2v.txt")
model = word2vec.Word2Vec(data, size=200, window=10, min_count= 5)

Price = model.most_similar(positive=[u'price'])
print(Price)
Product = model.most_similar(positive=[u'product'])
print(Product)

Apple = model.most_similar(positive=[u'apple'])
print(Apple)

Samsung = model.most_similar(positive=[u'samsung'])
print(Samsung)

vocabulary_lenght = len(model.wv.vocab)
print(vocabulary_lenght)

model.wv.save_word2vec_format("config/w2vmodel.csv")



# Cluter_Data = pd.read_csv("config/amazon-reviews-unlocked-mobile-phones.zip",nrows=6000)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem import WordNetLemmatizer
# import re
# Cluter_Data.columns

# from nltk.corpus import stopwords
# stop = set(stopwords.words('english'))
# from nltk.corpus import stopwords
# def remove_stopword(word):
#     return word not in words

# from nltk.stem import WordNetLemmatizer
# Lemma = WordNetLemmatizer()

# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english")

# Cluter_Data['NewReviews'] = Cluter_Data['Reviews'].str.lower().str.split()
# Cluter_Data['NewReviews'] = Cluter_Data['NewReviews'].apply(lambda x : [item for item in x if item not in stop])
# #Cluter_Data['NewReviews'] = Cluter_Data["NewReviews"].apply(lambda x : [stemmer.stem(y) for y in x])

# Cluter_Data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
# for line in lists]).strip() for lists in Cluter_Data['NewReviews']]
# print(Cluter_Data.columns)

# vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
# X = vectorizer.fit_transform(Cluter_Data['text_lem'].str.upper())

# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=5,init='k-means++',max_iter=200,n_init=1)

# km.fit(X)
# terms = vectorizer.get_feature_names()
# order_centroids = km.cluster_centers_.argsort()[:,::-1]
# for i in range(5):
#     print("cluster %d:" %i)
#     for ind in order_centroids[i,:10]:
#         print(' %s' % terms[ind])
#     print()