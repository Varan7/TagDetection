import re
import pickle as pkl
import numpy as np

def remove_punctuation(text):
	text = re.sub('[()-,.!@#$%^&*?]','',text)
	cleanr = re.compile('<.*?>')
	text = re.sub(cleanr,'',text)
	return(text)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
stop_words = set(stopwords.words('english'))

def remove_stopwords(text,stop_words):
	word_tokens = word_tokenize(text)
	filtered_text = [w.lower() for w in word_tokens if not w.lower() in stop_words]
	final_text = ' '.join(filtered_text)
	return(final_text)


with open('./new_text.txt','r') as fil:
	text = []
	for line in fil:
		line = line.rstrip().strip(' ')
		line = remove_punctuation(line)
		line = remove_stopwords(line,stop_words)
		text.append(line)

with open('./tags.txt','r') as f:
	tags = []
	for line in f:
		line = line.strip()
		line = remove_punctuation(line)
		line = remove_stopwords(line,stop_words)
		tags.append(line)




with open('./vectors_pretrained.txt','rb') as fil:
	vectors = pkl.load(fil)

with open('./vectors2_pretrained.txt','rb') as fil:
	vectors2 = pkl.load(fil)

with open('./vectors3_pretrained.txt','rb') as fil:
	vectors3 = pkl.load(fil)

with open('./words.txt','rb') as fil:
	words = pkl.load(fil)

vector_pretrained = {**vectors,**vectors2,**vectors3}

"""
vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}
    
vector_dim = len(vector_pretrained[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vector_pretrained.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v

W_norm = W
"""

tag_vectors = {}
for word in tags:

	word = remove_stopwords(word,stop_words)
	"""
	avg_tag = np.zeros(300,)
	count = 0
	for a in word.split(' '):
		if a in words:
			avg_tag += vector_pretrained[a]
			#avg_tag += W_norm[vocab[a]]
			count = count +1
	avg_tag = avg_tag/count
	tag_vectors[word] = avg_tag
	"""
	tag_vectors[word] = np.mean([vector_pretrained[a] for a in word.split(' ') if a in words], axis=0)
	

with open('./tag_vectors.txt','wb') as fil:
	pkl.dump(tag_vectors,fil)

text_vectors = {}
chat_vectors = {}
for word in text:
	"""
	avg_text = np.zeros(300,)
	count = 0
	"""
	to_mean = []
	for a in word.split(' '):
		if a in words:
			chat_vectors[a] = vector_pretrained[a]
			to_mean.append(chat_vectors[a])
			"""
			chat_vectors[a] = W_norm[vocab[a]]
			avg_text += W_norm[vocab[a]]
			count = count +1
			"""
	text_vectors[word] = np.mean(to_mean, axis=0)
	"""
	avg_text = avg_text/count
	text_vectors[word]=avg_text
	"""


with open('./text_vectors.txt','wb') as fil:
	pkl.dump(text_vectors,fil)

with open('./chat_vectors.txt','wb') as fil:
	pkl.dump(chat_vectors,fil)