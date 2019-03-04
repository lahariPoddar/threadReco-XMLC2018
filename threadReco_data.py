import os
from os.path import dirname, abspath
from keras.preprocessing.text import Tokenizer,one_hot
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
#from urlparse import urlparse
import re
import random


ROOT = os.getcwd()#dirname(dirname(abspath(__file__)))
DATA_DIR = ROOT + '/data/'
max_post_len = 30
EMBEDDING_DIM = 100
MAX_NB_WORDS = 50000
forum =  '<data_file_name>' #without the file extension

def read_file(fname):
	posts = []
	user_list = []
	with open(fname,'r',encoding='latin1') as f:
		reader = csv.DictReader(f, delimiter='\t')
		for row in reader:
			posts.append(row['postText'])
			user_list.append([int(x) for x in row['users'].split(',')])
	return posts,user_list

def get_embeddings_index():
	with open('glove.6B.{}d.txt'.format(EMBEDDING_DIM),encoding='latin1') as f:
		embeddings_index = {}
		for line in f:
			values = line.split(' ')
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	# print 'Found %s word vectors.' % len(embeddings_index)
	return embeddings_index

def process_data():

	file_path = DATA_DIR + forum+'.tsv'
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	
	posts_all, user_list_all = read_file(file_path)
	for i in range(0,len(posts_all)):
		t = posts_all[i]
		t = t.lower()
		t = ' '.join(re.sub("!"," _exmark_ ",t).split())
		t = ' '.join(re.sub("\?"," _qmark_ ",t).split())
		t = ' '.join(re.sub("can't","can not",t).split())
		t = ' '.join(re.sub("won't","will not",t).split())
		t = ' '.join(re.sub("ain't","is not",t).split())
		t = ' '.join(re.sub("n't "," not ",t).split())
		t = ' '.join(re.sub(r'http\S+', '_url_', t).split())
		posts_all[i] = t

	end = int(len(posts_all)*0.8)
	posts, user_list = posts_all[0:end], user_list_all[0:end]
	posts_test, user_list_test = posts_all[end:], user_list_all[end:]

	tokenizer.fit_on_texts(posts+posts_test)

	sequences_of_posts = tokenizer.texts_to_sequences(posts)
	sequences_of_posts = pad_sequences(sequences_of_posts,maxlen=max_post_len)
	posts = np.asarray(sequences_of_posts)
	
	sequences_of_test_posts = tokenizer.texts_to_sequences(posts_test)
	sequences_of_test_posts = pad_sequences(sequences_of_test_posts,maxlen=max_post_len)
	posts_test = np.asarray(sequences_of_test_posts)
	
	
	word_index = tokenizer.word_index

	count = 0
	tcount = 0 
	
	embeddings_index = get_embeddings_index()
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
			count += 1
		else:
			tcount += 1
			# embedding_matrix[i] = random.randint()
			for j in range(0,EMBEDDING_DIM):
				embedding_matrix[i][j] = random.randint(-20000,+20000)/10000.0
			# embedding_matrix[i] = 
	# print (count/float(count+tcount))*100
	del embeddings_index

	
	# print 'Found %s unique tokens.' % len(word_index)
	le = preprocessing.LabelEncoder()
	le.fit(np.hstack(user_list+user_list_test))
	
	le_y = [le.transform(u) for u in user_list]
	le_y_test = [le.transform(u) for u in user_list_test]
	enc = preprocessing.MultiLabelBinarizer()
	enc.fit(le_y+le_y_test)
	y = enc.transform(le_y)
	y_test = enc.transform(le_y_test)


	return {'embedding_matrix':embedding_matrix,
			'len_word_index':len(word_index),
			'x_train':posts,
			'x_test':posts_test,
			'y_train_list':le_y,
			'y_test_list':le_y_test,
			'y_train':y,
			'y_test':y_test
			}

d = process_data()

