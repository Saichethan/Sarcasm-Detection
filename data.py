import os
import csv
import numpy as np
import pandas as pd
import pickle

from embedding_as_service.text.encode import Encoder  



def parseJson(fname):
    for line in open(fname, 'r'):
        yield eval(line)

data = list(parseJson('Sarcasm_Headlines_Dataset.json'))


print("length: ", len(data))


print(data[0])

#is_sarcastic
#headline


def load_emb():

	sentences = []
	labels = []

	albert = []
	elmo = []
	xlnet = []
	
	bert = []
	w2v = []
	ft = []
	gv = []


	data = list(parseJson('Sarcasm_Headlines_Dataset.json'))	

	for i in range(len(data)):
		sentences.append(data[i]['headline'])
		labels.append(int(data[i]['is_sarcastic']))
	"""	
	print("Number of sentences & labels: ", len(sentences), len(labels))


	xl = Encoder(embedding='xlnet', model='xlnet_large_cased')
	xlnet_temp = xl.encode(texts=sentences, pooling='reduce_mean')

	
	el = Encoder(embedding='elmo', model='elmo_bi_lm')
	elmo_temp = el.encode(texts=sentences, pooling='reduce_mean') 
	
	al = Encoder(embedding='albert', model='albert_large')
	albert_temp = al.encode(texts=sentences, pooling='reduce_mean') 


	print("shape of xlnet, albert and elmo: ", xlnet_temp.shape, albert_temp.shape, elmo_temp.shape)


	for i in range(len(xlnet_temp)):
		xlvector = xlnet_temp[i]
		xlnet.append(xlvector)

	
	for i in range(len(elmo_temp)):
		elvector = elmo_temp[i]
		elmo.append(elvector)
	
	for i in range(len(albert_temp)):
		alvector = albert_temp[i]
		albert.append(elvector)

	
	elmo = np.asarray(elmo)
	albert = np.asarray(albert)
	xlnet = np.asarray(xlnet)

	print("shape of xlnet, albert and elmo: ", xlnet.shape, albert.shape, elmo.shape)
	
	
	
	np.ndarray.dump(xlnet, open('saved/xlnet.np', 'wb'))
	np.ndarray.dump(albert, open('saved/albert.np', 'wb'))
	np.ndarray.dump(elmo, open('saved/elmo.np', 'wb'))
	
	labels = np.asarray(labels)
	np.ndarray.dump(labels, open('saved/labels.np', 'wb'))

	"""

	print("Number of sentences & labels: ", len(sentences), len(labels))


	bte = Encoder(embedding='bert', model='bert_large_cased')
	bte_temp = bte.encode(texts=sentences, pooling='reduce_mean')

	
	w2ve = Encoder(embedding='word2vec', model='google_news_300')
	w2ve_temp = w2ve.encode(texts=sentences, pooling='reduce_mean') 
	
	fte = Encoder(embedding='fasttext', model='common_crawl_300')
	fte_temp = fte.encode(texts=sentences, pooling='reduce_mean') 


	#print("shape of xlnet, albert and elmo: ", xlnet_temp.shape, albert_temp.shape, elmo_temp.shape)


	for i in range(len(bte_temp)):
		btevector = bte_temp[i]
		bert.append(btevector)

	
	for i in range(len(w2ve_temp)):
		w2vevector = w2ve_temp[i]
		w2v.append(w2vevector)
	
	for i in range(len(fte_temp)):
		ftevector = fte_temp[i]
		ft.append(ftevector)

	
	bert = np.asarray(bert)
	w2v = np.asarray(w2v)
	ft = np.asarray(ft)

	#print("shape of xlnet, albert and elmo: ", xlnet.shape, albert.shape, elmo.shape)
	
	
	
	np.ndarray.dump(bert, open('saved/bert.np', 'wb'))
	np.ndarray.dump(w2v, open('saved/w2v.np', 'wb'))
	np.ndarray.dump(ft, open('saved/fasttext.np', 'wb'))

load_emb()
