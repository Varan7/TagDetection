import numpy as np

with open('./glove.6B.300d.txt','r') as f:
 	vectors = {}
 	vectors2 = {}
 	vectors3 = {}
 	words = []
 	count = 0
 	for line in f:
 		vals = line.rstrip().split(' ')
 		if(count<200000):
 			vectors[vals[0]] = np.array(list(map(float, vals[1:])))
 			words.append(vals[0])
 			count = count + 1
 		elif (count < 300000) :
 			vectors2[vals[0]] = np.array(list(map(float, vals[1:])))
 			words.append(vals[0])
 			count = count + 1
 		else :
 			vectors3[vals[0]] = np.array(list(map(float, vals[1:])))
 			words.append(vals[0])
 			count = count +1


import pickle as pkl
with open('./vectors_pretrained.txt','wb') as f:
	pkl.dump(vectors,f)

with open('./vectors2_pretrained.txt','wb') as f:
	pkl.dump(vectors2,f)

with open('./vectors3_pretrained.txt','wb') as f:
	pkl.dump(vectors3,f)

with open('./words.txt','wb') as f:
	pkl.dump(words,f)

