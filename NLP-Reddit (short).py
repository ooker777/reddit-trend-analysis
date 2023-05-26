# get the start time
import time
st = time.time()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import pickle
import sys
import os
import pickle
from gensim import models

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Logging is the verbose for Gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#plt.style.available # Style options
plt.style.use('fivethirtyeight')
sns.set_context("talk")

pd.options.display.max_rows = 99
pd.options.display.max_columns = 99
pd.options.display.max_colwidth = 99
#pd.describe_option('display') # Option settings

float_formatter = lambda x: "%.3f" % x if x>0 else "%.0f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
pd.set_option('display.float_format', float_formatter)

path = "data/all235.pkl"
df = pd.read_pickle(path)
df.info(memory_usage='Deep')

# ## Exploratory Data Analysis & Preprocessing

# Exploring data by length of .title or .selftext
df[[ True if 500 < len(x) < 800 else False for x in df.selftext ]].sample(3, replace=False)


run = False
path = 'data/gif'

# Run through various selftext lengths and save the plots of the distribution of the metric
# Gif visual after piecing all the frames together
while run==True:
    for i in range(500,20000,769):
        tempath = os.path.join(path, f"textlen{i}.png") # PEP498 requires python 3.6
        print(tempath)

        # Look at histogram of posts with len<i
        cuts = [len(x) for x in df.selftext if len(x)<i]

        # Save plot
        plt.figure()
        plt.hist(cuts, bins=30) #can change bins based on function of i
        plt.savefig(tempath, dpi=120, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()


# Bin Settings
def binSize(lower, upper, buffer=.05):
    bins = upper - lower
    buffer = int(buffer*bins)
    bins -= buffer
    print('Lower Bound:', lower)
    print('Upper Bound:', upper)
    return bins, lower, upper

# Plotting 
def plotHist(tmp, bins, title, xlabel, ylabel, l, u):
    plt.figure(figsize=(10,6))
    plt.hist(tmp, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(lower + l, upper + u)
    print('\nLocal Max %s:' % xlabel, max(tmp))
    print('Local Average %s:' % xlabel, int(np.mean(tmp)))
    print('Local Median %s:' % xlabel, int(np.median(tmp)))
    plt.savefig(title)

# Create the correct bin size
bins, lower, upper = binSize(lower=0, upper=175)

# Plot distribution of lower scores
tmp = df[[ True if lower <= x <= upper else False for x in df['score'] ]]['score']
plotHist(tmp=tmp, bins=bins, title='Lower Post Scores', xlabel='Scoring', ylabel='Frequency', l=5, u=5);


# Titles should be less than 300 charcters 
# Outliers are due to unicode translation
# Plot lengths of titles
tmp = [ len(x) for x in df.title ]
bins, lower, upper = binSize(lower=0, upper=300, buffer=-.09)

plotHist(tmp=tmp, bins=bins, title='Lengths of Titles', xlabel='Length', ylabel='Frequency', l=10, u=0);


# Slice lengths of texts and plot histogram
bins, lower, upper = binSize(lower=500, upper=5000, buffer=.011)
tmp = [len(x) for x in df.selftext if lower <= len(x) <= upper]

plotHist(tmp=tmp, bins=bins, title='Length of Self Posts Under 5k', xlabel='Length', ylabel='Frequency', l=10, u=0)
plt.ylim(0, 200);

# Anomalies could be attributed to bots or duplicate reposts


# Posts per Subreddit
# tmp = df.groupby('subreddit').nunique().sort_values(ascending=False)
# tmp = df.groupby('subreddit')['id'].nunique().sort_values(ascending=False)
tmp = df.groupby('subreddit')['id']
# tmp = df.nunique().sort_values(ascending=False)
# top = 100
# s = sum(tmp)
# print('Subreddits:', len(tmp))
# print('Total Posts:', s)
# print('Total Posts from Top %s:' % top, sum(tmp[:top]), ', %.3f of Total' % (sum(tmp[:top])/s))
# print('Total Posts from Top 10:', sum(tmp[:10]), ', %.3f of Total' % (sum(tmp[:10])/s))
# print('\nTop 10 Contributors:', tmp[:10])



# plt.figure(figsize=(10,6))
# plt.plot(tmp, 'go')
# plt.xticks('')
# plt.title('Top %s Subreddit Post Counts' % top)
# plt.xlabel('Subreddits, Ranked')
# plt.ylabel('Post Count')    
# plt.xlim(-2, top+1)
# plt.ylim(0, 2650);


path1 = 'data/origin.pkl'
#path2 = 'data/grouped.pkl'

# Save important data
origin_df = df.loc[:,['created_utc', 'subreddit', 'author', 'title', 'selftext', 'id']] \
              .copy().reset_index().rename(columns={"index": "position"})
print(origin_df.info())
origin_df.to_pickle(path1)

posts_df = origin_df.loc[:,['title', 'selftext']]
posts_df['text'] = posts_df.title + ' ' + df.selftext
#del origin_df

# To group the results later
def groupUserPosts(x):
    ''' Group users' id's by post '''
    return pd.Series(dict(ids = ", ".join(x['id']),                    
                          text = ", ".join(x['text'])))

###df = posts_df.groupby('author').apply(groupUserPosts) 
#df.to_pickle(path2)

df = posts_df.text.to_frame()

origin_df.sample(2).drop('author', axis=1)

def clean_text(df, text_field):
    '''
    Clean all the text data within a certain text column of the dataFrame.
    '''
    df[text_field] = df[text_field].str.replace(r"http\S+", " ")
    df[text_field] = df[text_field].str.replace(r"&[a-z]{2,4};", "")
    df[text_field] = df[text_field].str.replace("\\n", " ")
    df[text_field] = df[text_field].str.replace(r"#f", "")
    df[text_field] = df[text_field].str.replace(r"[\’\'\`\":]", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9]", " ")
    df[text_field] = df[text_field].str.replace(r" +", " ")
    df[text_field] = df[text_field].str.lower()
    
clean_text(df, 'text')

df.sample(3)


# For exploration of users
df[origin_df.author == '<Redacted>'][:3]

# User is a post summarizer and aggregator, added /r/tldr to the blocked list!


# Slice lengths of texts and plot histogram
bins, lower, upper = binSize(lower=500, upper=5000, buffer=.015)
tmp = [len(x) for x in df.text if lower <= len(x) <= upper]

plotHist(tmp=tmp, bins=bins, title='Cleaned - Length of Self Posts Under 5k', 
         xlabel='Lengths', ylabel='Frequency', l=0, u=0)
plt.ylim(0, 185);


# Download everything for nltk! ('all')
import nltk
# nltk.download() # (Change config save path)
nltk.data.path.append(r'C:\Users\ganuo\AppData\Roaming\nltk_data')


from nltk.corpus import stopwords

# "stopeng" is our extended list of stopwords for use in the CountVectorizer
# I could spend days extending this list for fine tuning results
stopeng = stopwords.words('english')
stopeng.extend([x.replace("\'", "") for x in stopeng])
stopeng.extend(['nbsp', 'also', 'really', 'ive', 'even', 'jon', 'lot', 'could', 'many'])
stopeng = list(set(stopeng))


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Count vectorization for LDA
cv = CountVectorizer(token_pattern='\\w{3,}', max_df=.30, min_df=.0001, 
                     stop_words=stopeng, ngram_range=(1,1), lowercase=False,
                     dtype='uint8')

# Vectorizer object to generate term frequency-inverse document frequency matrix
tfidf = TfidfVectorizer(token_pattern='\\w{3,}', max_df=.30, min_df=.0001, 
                        stop_words=stopeng, ngram_range=(1,1), lowercase=False,
                        sublinear_tf=True, smooth_idf=False, dtype='float32')

# ###### Tokenization is one of the most important steps in NLP, I will explain some of my parameter choices in the README. CountVectorizer was my preferred choice. I used these definitions to help me in the iterative process of building an unsupervised model.
# 
# ###### The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.
# 
# ###### Smooth = False: The effect of adding “1” to the idf in the equation above is that terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored.
# 
# ###### sublinear_tf = True: “l” (logarithmic), replaces tf with 1 + log(tf)



# Count & tf-idf vectorizer fits the tokenizer and transforms data into new matrix
cv_vecs = cv.fit_transform(df.text).transpose()
tf_vecs = tfidf.fit_transform(df.text).transpose()
pickle.dump(cv_vecs, open('data/cv_vecs.pkl', 'wb'))

# Checking the shape and size of the count vectorizer transformed matrix
# 47,317 terms
# 146996 documents
print("Sparse Shape:", cv_vecs.shape) 
print('CV:', sys.getsizeof(cv_vecs))
print('Tf-Idf:', sys.getsizeof(tf_vecs))


# IFF using a subset can you store these in a Pandas DataFrame/

#tfidf_df = pd.DataFrame(tf_vecs.transpose().todense(), columns=[tfidf.get_feature_names()]).astype('float32')
#cv_df = pd.DataFrame(cv_vecs.transpose().todense(), columns=[cv.get_feature_names()]).astype('uint8')

#print(cv_df.info())
#print(tfidf_df.info())


#cv_description = cv_df.describe().T
#tfidf_description = tfidf_df.describe().T

#tfidf_df.sum().sort_values(ascending=False)


# Explore the document-term vectors
#cv_description.sort_values(by='max', ascending=False)
#tfidf_description.sort_values(by='mean', ascending=False)


# ## Singular Value Decomposition (SVD)


# ## Latent Dirichlet Allocation (LDA)



run = False

passes = 85
if run==True:
    lda = models.LdaMulticore(corpus=cv_corpus, num_topics=30, id2word=id2word, passes=passes, 
                              workers=13, random_state=42, eval_every=None, chunksize=6000)


# Save model after your last run, or continue to update LDA
#pickle.dump(lda, open('data/lda_gensim.pkl', 'wb'))

# Gensim save
#lda.save('data/gensim_lda.model')
lda =  models.LdaModel.load('data/gensim_lda.model')



cv_corpus = pickle.load(open('data/cv_corpus.pkl','rb'))

# Transform the docs from the word space to the topic space (like "transform" in sklearn)
lda_corpus = lda[cv_corpus]

# Store the documents' topic vectors in a list so we can take a peak
lda_docs = [doc for doc in lda_corpus]


# Review Dirichlet distribution for documents
lda_docs[200]
# lda_docs[25000]


# Manually review the document to see if it makes sense! 
# Look back at the topics that it matches with to confirm the result!
df.iloc[200]


#bow = df.iloc[1,0].split()

# Print topic probability distribution for a document
#print(lda[bow]) #Values unpack error

# Given a chunk of sparse document vectors, estimate gamma:
# (parameters controlling the topic weights) for each document in the chunk.
#lda.inference(bow) #Not enough values

# Makeup of each topic! Interpretable! 
# The d3 visualization below is far better for looking at the interpretations.
lda.print_topics(num_words=10, num_topics=1)


# ## <a id='py'></a> pyLDAvis


# For quickstart, we can just jump straight to results

def loadingPickles():
    id2word = pickle.load(open('data/id2word.pkl','rb'))
    cv_vecs = pickle.load(open('data/cv_vecs.pkl','rb'))
    cv_corpus = pickle.load(open('data/cv_corpus.pkl','rb'))
    lda =  models.LdaModel.load('data/gensim_lda.model')
    return id2word, cv_vecs, cv_corpus, lda


import pyLDAvis.gensim

# Prepare the visualization
# Change multidimensional scaling function via mds parameter
# Options are tsne, mmds, pcoa 
# cv_corpus or cv_vecs work equally
id2word, _, cv_corpus, lda = loadingPickles()
viz = pyLDAvis.gensim.prepare(topic_model=lda, corpus=cv_corpus, dictionary=id2word, mds='mmds')

# Save the html for sharing!
pyLDAvis.save_html(viz,'data/viz.html')

# Interact! Saliency is the most important metric that changes the story of each topic.
pyLDAvis.display(viz)

# logging
et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')