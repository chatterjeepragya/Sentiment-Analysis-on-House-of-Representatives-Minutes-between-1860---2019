import whoosh.index as index
from whoosh.qparser import QueryParser
from whoosh import scoring 
from whoosh.fields import Schema, ID, TEXT
from whoosh.searching import Searcher
import nltk, re, os, glob
from nltk.sentiment import SentimentIntensityAnalyzer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

stemmer = SnowballStemmer("english")   # initiate snowball stemmer from nltk
ind = index.open_dir("indexdir") ## opening the collaborated files
analyzer = SentimentIntensityAnalyzer() ##initializing analyzer variable 

resultfile = open('Results.txt', 'w') ##opening file to write relevant text snippets into
user_query = input("Enter the name of an American political party:\n") ## accepting user query
q_parser = QueryParser("content", schema = ind.schema) 
parsed_query = q_parser.parse(user_query) ## parsing user query 
with ind.searcher(weighting=scoring.TF_IDF()) as searcher: 
    results = searcher.search(parsed_query, sortedby="title", limit=None)
    print("Your political party has been mentioned in " + str(len(results)) +"historic documents and the following is the sentiment analysis and frequency distribution:\n")
    seq = 0
    for hit in results: 
        seq = seq +1 
        print (seq, hit['path']) 
        print (analyzer.polarity_scores(hit.highlights("content", top=1)))
        resultfile.write(hit.highlights("content", top=1))
resultfile.close()
print("\n") 

#Function to tokenize raw text 
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters 
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems # output is a list of tokens
       
df = open('Results.txt', "r")
lines = df.readlines()
df.close()

# remove /n at the end of each line
for index, line in enumerate(lines):
      lines[index] = line.strip()
        
tfidf_vectorizer = TfidfVectorizer(max_df=8, min_df=1, tokenizer=tokenize_and_stem)
tfidf_matrix = tfidf_vectorizer.fit_transform(lines)
k = 6
km = KMeans(n_clusters=k)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print("Frequency of the clusters is as follows:\n")
print(list(clusters))
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster", i, ":")
    print([terms[ind] for ind in order_centroids[i, :15]])
