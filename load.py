import os
import csv
import numpy as np
import pandas as pd
import pickle




def load_data():
	albert = pickle.load(open('saved/albert.np', 'rb'))
	elmo = pickle.load(open('saved/elmo.np', 'rb'))
	xlnet = pickle.load(open('saved/xlnet.np', 'rb'))

	bert = pickle.load(open('saved/bert.np', 'rb'))
	w2v = pickle.load(open('saved/w2v.np', 'rb'))
	ft = pickle.load(open('saved/fasttext.np', 'rb'))

	y = pickle.load(open('saved/labels.np', 'rb'))

	return albert, bert, elmo, ft, w2v, xlnet, y 
	



