from threadReco_data import process_data, EMBEDDING_DIM, max_post_len
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, multiply, concatenate, add
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda,Permute, Flatten, Dense, Dropout, Activation, Reshape, RepeatVector, Masking
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau,LambdaCallback
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from os.path import dirname, abspath
from keras.preprocessing.sequence import pad_sequences
from os import listdir
import numpy as np
import h5py, pickle
from random import randint, choice, shuffle, sample
from sys import argv
from keras.layers.advanced_activations import PReLU
from keras.utils import plot_model
import random
from keras.layers.merge import Multiply
from sklearn.utils import class_weight
from keras.layers.convolutional import  Conv1D,MaxPooling1D

max_len = max_post_len
gru_dim = 128
bigru_dim = 2*gru_dim
out_size = max_len*bigru_dim
clusters = 100
def sum_att(x):
	return K.sum(x,axis=1)

def stacking(x):
        return K.stack(x,axis=1)

def init_model(num_users,data=None):
	print('Compiling model...')
	
	embedding_matrix, len_word_index = data
	print('building the model...')
	inp = Input(shape = (max_len,),name="input")
		
	x = Sequential()
	x.add(Embedding(len_word_index + 1,
							EMBEDDING_DIM,
							#mask_zero=True,
							input_length=max_len,
							weights=[embedding_matrix],
						) )
	x.add(Bidirectional(GRU(gru_dim, return_sequences = True)))
	x.add(Bidirectional(GRU(gru_dim, return_sequences = True)))
	
	post = x(inp)
		
	y = Sequential()
	y.add(Flatten(input_shape=(max_len,bigru_dim)))
	y.add(Dense(max_len,activation='softmax', input_shape=(out_size,)))
	y.add(RepeatVector(bigru_dim))
	y.add(Permute((2,1)))
	

	posts = []
	for i in range(0,clusters): 
		post_att = y(post)
		post_i = Multiply()([post_att,post])
		post_i = Lambda(sum_att,output_shape=(bigru_dim,))(post_i)
		posts.append(post_i)

	posts = Lambda(stacking,output_shape=(clusters,bigru_dim,))(posts)

	out = TimeDistributed(Dense(1,name="dense1",activation='tanh')) (posts)
	out = Dropout(0.3, name='drop4') (out)
	out = Flatten()(out)

	output = Dense(num_users, activation='sigmoid') (out)
	model = Model(inputs=[inp], outputs=[output])
	print(y.summary())
	return model


def get_data():
	data = process_data()
	return data

def eval(x_test,y_test_list,model):
	kvals = [5,10,30,50,100]
	recall_k = []
	ndcg_k = []
	rr = []
	y_true_list = y_test_list
	y_pred = model.predict([x_test])
	for i in range(0,len(y_true_list)):
		y_t = y_true_list[i]
		y_t_len = len(y_t)
		y_p_index = np.flip(np.argsort(y_pred[i]),0)
		for i in range(0,len(y_p_index)):
			if y_p_index[i] in y_t:
				rr.append(1/float(i+1))
				break
		idcg = np.sum([1.0/np.log2(x+2) for x in range(0,y_t_len)])
		r = []
		ndcg = []
		for k in kvals:
			correct = len(np.intersect1d(y_t,y_p_index[0:k]))
			r.append(correct/float(y_t_len))
			dcg = 0
			for i in range(0,k):
				if y_p_index[i] in y_t:
					dcg = dcg + 1.0/np.log2(i+2)
			ndcg.append(dcg/idcg)
		ndcg_k.append(ndcg)
		recall_k.append(r)
		

	print(" mrr = "+str(np.mean(rr)))
	recall = np.mean(recall_k,axis=0)*100
	ndcg = np.mean(ndcg_k,axis=0)*100
	for i in range(0,len(kvals)):
		print(" recall@"+str(kvals[i])+"= "+str(recall[i]))
		print(" ndcg@"+str(kvals[i])+"= "+str(ndcg[i]))

def recall(y_true, y_pred):
	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
	return c1/c3

def main(args):
	data = get_data()
	
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']
	
	x_train = data['x_train']
	x_test = data['x_test']
	
	y_train_list = data['y_train_list']
	y_test_list = data['y_test_list']
	y_train = data['y_train']
	y_test = data['y_test']
	
	num_users = np.array(y_test).shape[1]
	print("y_train_shape: "+str(y_train.shape))
	print("y_test_shape: "+str(y_test.shape))
	print("num_users: "+str(num_users))
	rmsprop = RMSprop(lr=0.003)
	adam = Adam(lr=0.003)

	class_weights = [1.,10.] #class_weight.compute_class_weight('balanced', [0,1], y_train.flatten())
	print("Class Weights: ",class_weights) 	
	model = init_model(num_users,data=[embedding_matrix, len_word_index])
	model.compile(optimizer='adam',
				  metrics= ['acc'],
				  loss='binary_crossentropy')
	metric_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: eval(x_test,y_test_list,model))					
	model.fit([x_train],[y_train],
			validation_data=([x_test],[y_test]),callbacks=[metric_callback]
			,class_weight=[class_weights]
			,batch_size=64,epochs=20,verbose=1)

if __name__ == '__main__':
	main(argv[1:])

