
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
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import io

stopEnglish = stopwords.words('english')
stopGerman = stopwords.words('german')


#remove !"#$%&()*+,-./:;<=>?@[\\]^_`{|}~
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


#remove german (firstly) and english stopwords.
def remove_mystopwords(sentence):
    tokens = sentence.split(" ")
    tokens_filtered = [word for word in tokens if not word in stopGerman]
    tokens_filteredTwice = [word for word in tokens_filtered if not word in stopEnglish]
    return (" ").join(tokens_filteredTwice)


df = pd.read_json('GermanFakeNC.json')
df.to_csv('GermanFakeNC.csv', index = None)

lines = sum(1 for line in open('GermanFakeNC.csv'))

data = pd.read_csv("GermanFakeNC.csv")
dataText = pd.read_csv("GermanFakeNC_Texts.csv") 		#here will be stored the texts from URLs.
dataRating = pd.read_csv("GermanFakeNC_RateLength.csv") #here will be stored their targets-lengths.


for i in range(0,lines-1):

	try:
		htmlString = get(data["URL"][i]).text
		html = BeautifulSoup(htmlString, 'lxml')
		entries = html.find_all( {'class':'post-content entry-content', 'p':True})  # gets html code
		text = [e.get_text() for e in entries]  # seperates text from html elements such as <div> etc.

		if len(text) > 1:
		    text = [' '.join(text)]
		else: 
			data.drop(i,axis=0,inplace=True)
			continue;

		#data pre-process:
		text = [x.replace('\n', '') for x in text]
		text = [x.replace('\t', '') for x in text]
		newtext = text[0]
		newtext = newtext.lower()
		newtext = punctuation_removal(newtext)
		newtext = remove_mystopwords(newtext)

		data.at[i, 'URL'] =  newtext		#update value
		length = (len(newtext.split()))
		data.at[i, 'length'] = length

		#2 classes: 0 => real , 1 => fake
		if(data['Overall_Rating'][i] < 0.5 ):
			data.at[i, 'Overall_Rating'] = int(1)
		else:
			data.at[i, 'Overall_Rating'] = int(0)

		dataRating.at[i,'Rating'] = data['Overall_Rating'][i]
		dataText.at[i,'text'] = data['URL'][i]


	except:
		data.drop(i,axis=0,inplace=True)
		print("do not succeed opening website")
		continue


dataRating['length'] = data['length']

dataText.to_csv("GermanFakeNC_Texts.csv", index = False)
dataRating.to_csv("GermanFakeNC_RateLength.csv", index = False)

