import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import html2text
from requests import get
import urllib.request 
from inscriptis import get_text
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import html2text
import trafilatura
from lxml import html
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import flair
import csv
from flair.embeddings import WordEmbeddings
from tqdm import tqdm
from gensim.models import FastText
import io



glove_embedding = WordEmbeddings('de')


#load german "vocabulary" and correlations
def load_fasttext():
        print('loading word embeddings...')
        embeddings_index = {}
        f = open('dewiki.txt',encoding='utf-8')
        for line in tqdm(f):
	        values = line.strip().rsplit(' ')
	        word = values[0]
	        coefs = np.asarray(values[1:], dtype='float32')
	        embeddings_index[word] = coefs
        f.close()
        print('found %s word vectors' % len(embeddings_index))
    
        return embeddings_index



#	main program	#

data = pd.read_csv("GermanFakeNC_Texts.csv", lineterminator='\n') 
dataRate = pd.read_csv("GermanFakeNC_RateLength.csv", lineterminator='\n')

dataRate['length'] = dataRate['length'].apply(np.int64)

#tokenizing the text => converting the words, letters into counts or numbers.
maxFeatures = max(dataRate['length'])
tokenizer = Tokenizer(num_words=maxFeatures, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(texts=data['text'])

word_index = tokenizer.word_index
vocab_size = len(word_index)
X = tokenizer.texts_to_sequences(texts=data['text'])

#now applying padding to make them even shaped.
#this function transforms a list of num_samples sequences (lists of integers) into a 2d Numpy array of shape
#(num_samples, num_timesteps).
X = pad_sequences(sequences=X, maxlen=maxFeatures, padding='pre')  



embeddings_index = load_fasttext()

embeddings_matrix = np.zeros((vocab_size + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector



#	prepare data (train & test) for split 	#
y = dataRate['Rating'].values 
data['Overall_Rating'] = dataRate['Rating']
test_data = data
test = data


tokenizer.fit_on_texts(texts=test_data['text'])
test_text = tokenizer.texts_to_sequences(texts=test_data['text'])
test_text = pad_sequences(sequences=test_text, maxlen=maxFeatures, padding='pre')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)