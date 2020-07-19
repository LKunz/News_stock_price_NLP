###############################################################################
#                   ADVANCED DATA ANALYSIS 2019 - PROJECT
###############################################################################

# This code cleans the data and implements divers algorithms.
# For data retrieving, please see the retrieving code (retrieve.py).

#------------------------------------------------------------------------------
#                                 PARAMETERS
#------------------------------------------------------------------------------

# In this section one can set the different parameters to run the code

# Point in time (min) after which the news has broken to calculate returns
time_range = 5
# headline: 3, 5 (basecase), 10, 30
# story: 5, 10 (basecase), 20, 30

# Number of most frequent words to report (see: DESCRIPTIVE STATISTICS)
num = 20

# Content one wants for classification (should be 'story' or 'headline')
classification = 'headline'

# If number of words in news (cleaned) < word_lim, the news is deleted. 
# We thus avoid noise with news that are not long enough
word_limit = 0 # 20 for story, 0 for headline

# Raise error if false content
if classification not in ['headline', 'story']:
    raise Exception("Error : classification is neither 'story' nor 'headline'")


#------------------------------------------------------------------------------
#                           IMPORT AND CLEAN DATA
#------------------------------------------------------------------------------

# Import packages
import pandas as pd
import seaborn as sns
import collections
import string
from scipy.stats import skew, kurtosis
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
import math
import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import warnings
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from keras.utils.np_utils import to_categorical
import tensorflow as tf
np.random.seed(1995)
import tensorflow_hub as hub
import os
import re

# Initialize timer
start = datetime.datetime.now()

# Import news and prices data from cvs files
news = pd.read_csv('News.csv', index_col=0)
prices = pd.read_csv('prices.csv', parse_dates=[['<DATE>', '<TIME>']])
prices.columns = ['date', 'ticker', 'per', 'open', 
                  'high', 'low', 'close', 'volume']
prices.index = prices['date']
prices = prices.drop(['date'], axis=1)

# Rename one column in news
col = ['versionCreated', 'headline', 'storyId', 'sourceCode', 'story']
news.columns = col

# Compute return time_range minutes after news is published
def get_return(time, time_range):
    sTime_str = time
    # Convert string into time format
    try:
        sTime = datetime.datetime.strptime(sTime_str, 
                                           '%Y-%m-%d %H:%M:%S.%f+00:00')
    except:
        pass
    try:
        sTime = datetime.datetime.strptime(sTime_str, 
                                           '%Y-%m-%d %H:%M:%S+00:00')
    except:
        pass
    # Set seconds and micro seconds to zero
    sTime = sTime.replace(second=0,microsecond=0)
    # Convert UTC to New York time
    sTime -= datetime.timedelta(hours=4)
    # Compute return (in %)
    try:
        t0 = prices.iloc[prices.index.get_loc(sTime),2] # open price to be before the news is broken
        ret = ((prices.iloc[prices.index.get_loc((sTime + datetime.timedelta(minutes=time_range))),2]/(t0)-1)*100) 
    except:
        ret = np.nan
   
    return pd.Series([ret, sTime])

news['return'] = np.nan
news['date_time_NY'] = np.nan

news[['return', 'date_time_NY']] = news['versionCreated'].apply(get_return, time_range=time_range)

# Delete all news for which no return is available (outside trading hours)
print('Number of news imported: ' + str(len(news)))
print('Total number of words imported:')
print(news[classification].apply(lambda x: len(x.split())).sum())
news.dropna(inplace=True)
print('Number of news inside trading hours: ' + str(len(news)))
print('Total number of words inside trading hours:')
print(news[classification].apply(lambda x: len(x.split())).sum())

# Clean content
def clean_text(text, word_limit, stemming=True):
    """ Clean text 
        
        Parameters:
            text (str) : text to clean
            stemming (bool) : if we want to stem the words
            
        Return:
            cleanned text (str)
      
    """
    # Choose options for cleanning
    stop_words = set(stopwords.words('english')) 
    table = str.maketrans('', '', string.punctuation)
    porter = PorterStemmer() 

    # HTML decoding (in case it is)
    text = BeautifulSoup(text, "lxml").text
    # Lowercase text
    text = text.lower()
    # Split into words 
    tokens = word_tokenize(text)
    # Remove punctuation 
    stripped = [w.translate(table) for w in tokens]
    # Remove all tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # Remove all stopwords 
    words = [w for w in words if not w in stop_words]
    # Stemming words
    if stemming:
        stemmed = [porter.stem(word) for word in words] # we can disable it to test
    else:
        stemmed = words
    # Remove news that are too small
    if len(stemmed) < word_limit:
        stemmed = ''
    # Gather words
    sentence = ' '.join(stemmed)
    # Make all empty content nan (for removing)
    if sentence == '' or sentence == 'error':
        sentence = np.nan
    else:
        pass
    
    return sentence

news['clean_' + classification] = news[classification].apply(clean_text, word_limit=word_limit)

# Remove empty instances
news.dropna(inplace=True)

# Print one instance
print(news.iloc[100])

# Print total number of news after cleaning
print('Total number of news after cleanning: ' + str(len(news)))

# Print total number of words after cleanning
print('Total number of words after cleaning:')
print(news['clean_' + classification].apply(lambda x: len(x.split())).sum())  
 
# Make story non html format - not needed for text
def make_text(message):
     return BeautifulSoup(message,"lxml").text
 
if classification == 'story':
    news[classification] = news[classification].apply(make_text)

    
#------------------------------------------------------------------------------
#                           DESCRIPTIVE STATISTICS
#------------------------------------------------------------------------------

# In this section we try to get a first idea of what the data looks like
    
def PlotWords(wordslist, plot=True):
    """ Generate histogram of number of words versus word frequency
    
    Parameters:
        wordslist (list or array) : list of words
        plot (bool) : if True generate automatically scatter plot
    
    Return:
        word_count (dic) : number of occurence of each word in wordslist
    """
    
    # Get number of occurrence of each word in wordslist
    word_count = {}
    for word in wordslist:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
     
    # Get frenquencies of number of words
    word_freq = {}   
    for value in word_count.values():
        if value in word_freq:
            word_freq[value] += 1
        else:
            word_freq[value] = 1
    
    # Plot 
    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(word_freq.keys(), word_freq.values(), s=5, marker='o', color='black')
    plt.xlabel('Word frequency')
    plt.ylabel('Number of words')
    plt.xscale('log')
    plt.yscale('log')
    #plt.savefig('LateX/headline_clean_freq.png')
    plt.show()
    plt.close()
    
    return word_count

# Generate plots of words and print words that occur the most in clean news
words = []
X = news['clean_' + classification]
for i in X:
    tmp = i.split()
    words += tmp
dic = PlotWords(words)
word_counter = collections.Counter(dic)
for word, count in word_counter.most_common(num):
    print(word, ": ", count)
  
# Generate plots of words and print words that occur the most in raw news
words = []
X = news[classification]
for i in X:
    tmp = i.split()
    words += tmp
dic = PlotWords(words)
word_counter = collections.Counter(dic)
for word, count in word_counter.most_common(num):
    print(word, ": ", count)
    
# Plot distribution of returns
X = news['return'].values
sns.set_style('white')
sns.distplot(X, bins=100, kde=False)
plt.xlabel('Return (%)')
plt.ylabel('Frequency')
#plt.savefig('LateX/returns.png')
plt.show()
plt.close()

# Compute moments of returns
mean = np.mean(X)
print ('Mean of returns : ' + str(mean))
sigma = np.std(X)
print ('Standard deviation of returns : ' + str(sigma))
skew = skew(X)
print ('Skewness of returns : ' + str(skew))
kurt = kurtosis(X)
print ('Kurtosis of returns : ' + str(kurt))

# Histogram of length of news in words (raw news)
length = []
for message in news[classification].values:
    length.append(len(message.split()))
plt.hist(length, bins=30)
plt.title(classification)
plt.xlabel('Length of news (number of words)')
plt.ylabel('Frequency')
plt.show()
plt.close()

# Histogram of length of news in words (clean news)
length = []
for message in news['clean_' + classification].values:
    length.append(len(message.split()))
plt.hist(length, bins=20)
plt.xlabel('Length of news (number of words)')
plt.ylabel('Frequency')
#plt.savefig('LateX/headline_clean_length.png')
plt.show()
plt.close()

#------------------------------------------------------------------------------
#                              SENTIMENT ANALYSIS
#------------------------------------------------------------------------------

# In this section we perform a sentiment analysis on the content of the news

# Create new DataFrame to work with
SAnews = news[['headline', 'story', 'return']]

# Add three new columns 
SAnews['polarity'] = np.nan
SAnews['subjectivity'] = np.nan
SAnews['score'] = np.nan

# Determine Polarity, Subjectitivty and Score
def get_score(message):
    sentA = TextBlob(message)
    polarity = sentA.sentiment.polarity
    if polarity > 0.6:
        score = 'very positive'
    elif polarity > 0.2:
        score = 'positive'
    elif polarity > -0.2:
        score = 'neutral'
    elif polarity >= -0.6:
        score = 'negative'
    else:
        score = 'very negative'
        
    return score

def get_polarity(message):
    sentA = TextBlob(message)
    return sentA.sentiment.polarity

def get_subjectivity(message):
    sentA = TextBlob(message)
    return sentA.sentiment.subjectivity

SAnews['score'] = SAnews[classification].apply(get_score)
SAnews['polarity']  = SAnews[classification].apply(get_polarity)
SAnews['subjectivity'] = SAnews[classification].apply(get_subjectivity)

groups = SAnews.groupby(['score']).mean()

# Print one instance
print(SAnews.iloc[0])

# Print results
print(groups)


#------------------------------------------------------------------------------
#                    DATA PREPROCESSING FOR CLASSIFICATION
#------------------------------------------------------------------------------

# Create new dataframe for classification
CLnews = news[['headline', 'story', 'clean_' + classification, 'return']]

# Creation of output y = {-1,0,1}
def get_output(ret):
    if ret > 0.0:
        return 1
    elif ret <= 0.0 and ret >= -0.0:
        return np.nan # np.nan for two classes clasification
    else:
        return 0

CLnews['returnClass'] = np.nan
CLnews['returnClass'] = CLnews['return'].apply(get_output)

CLnews.dropna(inplace=True) # drop returns not in binary threshold

# Plot return classes
plt.hist(CLnews['returnClass'], color='green')
plt.title('Histogram of the return classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
plt.close()

# Print one instance
print(CLnews.iloc[76])


#------------------------------------------------------------------------------
#                                NAIVE BAYES
#------------------------------------------------------------------------------

print('Naive Bayes')

# Select input and output
X = CLnews['clean_' + classification].values # clean data
y = CLnews['returnClass'].values

# Bag of Words
tfidfconverter = TfidfVectorizer(max_features=None, min_df=10, max_df=0.8)
X_tfid = tfidfconverter.fit_transform(X).toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfid, y, test_size=0.2,
                                                    random_state=1995)

# Fit model
nb = MultinomialNB(alpha=1)
nb.fit(X_train, y_train)

# Test quality of the model
predicted_y = nb.predict(X_train)
print(confusion_matrix(y_true=y_train, y_pred=predicted_y))
print(accuracy_score(y_true=y_train, y_pred=predicted_y))

y_predicted = nb.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=y_predicted))
print(accuracy_score(y_true=y_test, y_pred=y_predicted))

# Print class priors
for c in range(0, len(nb.classes_)):
    print('Class: ' + str(c))
    print(str(math.exp(nb.class_log_prior_[c])))

# Find best/worst words (only for binary output)
up = {}
down = {}
for word in tfidfconverter.get_feature_names():
    wordTr = tfidfconverter.transform([word]).toarray()
    classTMP = nb.predict(wordTr)
    if classTMP == 1:
        up[word] = max(nb.predict_proba(wordTr)[0])
    else:
        down[word] = max(nb.predict_proba(wordTr)[0])
        
# Print 10 best/worst words
print('Best words')
word_counter = collections.Counter(up)
for word, proba in word_counter.most_common(10):
    print(word, ": ", proba)

print('Worst words')
word_counter = collections.Counter(down)
for word, count in word_counter.most_common(10):
    print(word, ": ", count)
    
# Test prediction with what you want
fakenews = 'Apple is the best company in the world!'
fakenews = clean_text(fakenews, word_limit=0)
fakenewsBoW = tfidfconverter.transform([fakenews]).toarray()
print(nb.predict(fakenewsBoW))
print(nb.predict_proba(fakenewsBoW))
print(nb.classes_)

# Test prediction with what you want
fakenews = 'The shareholders are losing attractive money'
fakenews = clean_text(fakenews, word_limit=0)
fakenewsBoW = tfidfconverter.transform([fakenews]).toarray()
print(nb.predict(fakenewsBoW))
print(nb.predict_proba(fakenewsBoW))
print(nb.classes_)

# Test prediction with what you want
fakenews = 'Investors are expecting share buybacks'
fakenews = clean_text(fakenews, word_limit=0)
fakenewsBoW = tfidfconverter.transform([fakenews]).toarray()
print(nb.predict(fakenewsBoW))
print(nb.predict_proba(fakenewsBoW))
print(nb.classes_)
    

#------------------------------------------------------------------------------
#                                   SVM
#------------------------------------------------------------------------------

print('SVM')

# Select input and output
X = CLnews['clean_' + classification].values # clean data
y = CLnews['returnClass'].values

# Bag of Words
tfidfconverter = TfidfVectorizer(max_features=None, min_df=10, max_df=0.8)
X_tfid = tfidfconverter.fit_transform(X).toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfid, y, test_size=0.2,
                                                    random_state=1995)

# Fit model
SVM = svm.SVC(C=1, kernel='linear', probability=True)
SVM.fit(X_train, y_train)

# Test quality of the model
predicted_y = nb.predict(X_train)
print(confusion_matrix(y_true=y_train, y_pred=predicted_y))
print(accuracy_score(y_true=y_train, y_pred=predicted_y))

predictions_svm = SVM.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=predictions_svm))
print(accuracy_score(y_true=y_test, y_pred=predictions_svm))

# Find best/worst words
up = {}
down = {}
for word in tfidfconverter.get_feature_names():
    wordTr = tfidfconverter.transform([word]).toarray()
    classTMP = SVM.predict(wordTr)
    if classTMP == 1:
        up[word] = max(SVM.predict_proba(wordTr)[0])
    else:
        down[word] = max(SVM.predict_proba(wordTr)[0])
        
# Print 10 best/worst words
print('Best words')
word_counter = collections.Counter(up)
for word, proba in word_counter.most_common(10):
    print(word, ": ", proba)

print('Worst words')
word_counter = collections.Counter(down)
for word, count in word_counter.most_common(10):
    print(word, ": ", count)
    
# Test prediction with what you want
fakenews = 'Apple is the best company in the world!'
fakenews = clean_text(fakenews, word_limit=0)
fakenewsBoW = tfidfconverter.transform([fakenews]).toarray()
print(SVM.predict(fakenewsBoW))
print(SVM.predict_proba(fakenewsBoW))
print(SVM.classes_)

# Test prediction with what you want
fakenews = 'The shareholders are losing attractive money'
fakenews = clean_text(fakenews, word_limit=0)
fakenewsBoW = tfidfconverter.transform([fakenews]).toarray()
print(SVM.predict(fakenewsBoW))
print(SVM.predict_proba(fakenewsBoW))
print(SVM.classes_)

# Test prediction with what you want
fakenews = 'Investors are expecting share buybacks'
fakenews = clean_text(fakenews, word_limit=0)
fakenewsBoW = tfidfconverter.transform([fakenews]).toarray()
print(SVM.predict(fakenewsBoW))
print(SVM.predict_proba(fakenewsBoW))
print(SVM.classes_)

"""
#------------------------------------------------------------------------------
#                                 NEURAL NETS
#------------------------------------------------------------------------------

# In this section we apply a deep learning algortihm
# (This section may be developped further in future projects)

# Create new dataframe
DDNnews = CLnews[[classification, 'return', 'returnClass']]

# Split
train_df, test_df = train_test_split(DDNnews, test_size=0.2, random_state=1995)

# Install TF-Hub.
#!pip install tensorflow-hub

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["returnClass"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["returnClass"], shuffle=False)

# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["returnClass"], shuffle=False)

# Embedding from source
embedded_text_feature_column = hub.text_embedding_column(
    key=classification, 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

# Embed features
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100], #500, 100
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.005)) #0.003

# Train the model
estimator.train(input_fn=train_input_fn, steps=1000);

# Compute accuracy of both train and test sets
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))

# Define function for later use
def get_predictions(estimator, input_fn):
  return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

LABELS = [
    "negative", "positive"
]

## Create a confusion matrix on training data.
#with tf.Graph().as_default():
#  cm = tf.confusion_matrix(train_df["returnClass"], 
#                           get_predictions(estimator, predict_train_input_fn))
#  with tf.Session() as session:
#    cm_out = session.run(cm)
#
## Normalize the confusion matrix so that each row sums to 1.
#cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]
#
#sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS);
#plt.xlabel("Predicted");
#plt.ylabel("True");

# Create a confusion matrix on test data
with tf.Graph().as_default():
  cm = tf.confusion_matrix(test_df["returnClass"], 
                           get_predictions(estimator, predict_test_input_fn))
  with tf.Session() as session:
    cm_out = session.run(cm)

# Normalize the confusion matrix so that each row sums to 1
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS);
plt.xlabel("Predicted");
plt.ylabel("True");

# Play with modules
def train_and_evaluate_with_module(hub_module, train_module=False):
  embedded_text_feature_column = hub.text_embedding_column(
      key=classification, module_spec=hub_module, trainable=train_module)

  estimator = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      feature_columns=[embedded_text_feature_column],
      n_classes=2,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.0005))

  estimator.train(input_fn=train_input_fn, steps=1000)

  train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

  training_set_accuracy = train_eval_result["accuracy"]
  test_set_accuracy = test_eval_result["accuracy"]

  return {
      "Training accuracy": training_set_accuracy,
      "Test accuracy": test_set_accuracy
  }

# Save results
results = {}
results["nnlm-en-dim128"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/nnlm-en-dim128/1")
results["nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/nnlm-en-dim128/1", True)
results["random-nnlm-en-dim128"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/random-nnlm-en-dim128/1")
results["random-nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/random-nnlm-en-dim128/1", True)
"""

#------------------------------------------------------------------------------
#                               POSTPROCESSING
#------------------------------------------------------------------------------

# Compute elapsed time
time_elapsed = (datetime.datetime.now() - start)
print('Time elapsed "%02d:%02d:%02d" % (h, m, s) {}'.format(time_elapsed))

# Copyright
print('Copyright (c) 2019 DAVIDLUCBENE (DLB) hedgefund.\nAll Rights Reserved.')












    


        
    
    
    
     