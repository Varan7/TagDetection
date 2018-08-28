import re
import numpy as np
import pickle as pkl




def cosine_similarity(u,v):
    dot = np.dot(u,v)
    norm_u = np.sqrt(np.sum(u*u))
    norm_v = np.sqrt(np.sum(v*v))
    cos_sim = dot/(norm_u*norm_v)
    return cos_sim



def text_to_average(text,W_norm,vocab):
 	avg = np.zeros(300,)
 	count = 0
 	for w in text.split():
 		if w in vocab.keys():
 			avg +=W_norm[vocab[w]]
 			count = count +1
 	avg = avg/count
 	
 	return avg

 


with open('./tag_vectors.txt','rb') as fil :
	tag_vectors = pkl.load(fil)

with open('./text_vectors.txt','rb') as fil:
	text_vectors = pkl.load(fil)

with open('./chat_vectors.txt','rb') as fil:
	chat_vectors = pkl.load(fil)

import operator
count2 = 1
for text,text_vector in text_vectors.items() :
	length = len(text.split(' '))
	if(length>12):
		print(text)
		count2 = count2 + 1
		similarity_value = {}
		similarity = [None]*43
		count = 0
		count1 = 0
		for tag,tag_vector in tag_vectors.items():
			similarity_value[tag] = cosine_similarity(text_vector,tag_vector)
			
	
		for k,v in similarity_value.items():
			similarity[count1] = v
			count1 = count1 + 1
		similarity.sort(reverse = True)
		while(count<6):
			for k,v in similarity_value.items():
				if v == similarity[count]:
					if (k == 'work'):
						if(v>0.55):
							print(k,v)
							count = count + 1
						else:
							count = count + 1
					elif(k == 'wasting time'):
						if(v>0.5):
							print(k,v)
							count = count + 1
						else:
							count = count + 1
					else:
						print(k,v)
						count = count + 1



	else:
		print(text)
		chat_max = {}
		for tag,tag_vector in tag_vectors.items():
			chat_max[tag] = -1
		for word in text.split(' '):
			word_similarity = {}
			max_value = -1
			second_max_value = -1
			for tag,tag_vector in tag_vectors.items():
				word_similarity[tag] = cosine_similarity(chat_vectors[word],tag_vector)
				if (word_similarity[tag]>max_value):
					max_value = word_similarity[tag]
					current_max_tag = tag
			
			if(chat_max[current_max_tag]<max_value):
				chat_max[current_max_tag] = max_value
			for tag,tag_vector in tag_vectors.items():
				if (tag!=current_max_tag and word_similarity[tag]>second_max_value):
					second_max_value = word_similarity[tag]
					second_max_tag = tag
			if(chat_max[second_max_tag]<second_max_value):
				chat_max[second_max_tag] = second_max_value
	
		count8 = 0
		chat_similarity = [None]*43
		
		for key,value in chat_max.items():
			chat_similarity[count8] = value
			count8 = count8 + 1
		chat_similarity.sort(reverse = True)
		count9 = 0
		while(count9<6):
			for k,v in chat_max.items():
				if (v==chat_similarity[count9]):

					if(chat_similarity[count9]==-1):
						count9 =6
					elif(k =='work'):
						if(v>0.55):
							print(k,v)
							count9 = count9 + 1
						else:
							count9 = count9 + 1
					
					elif(k == 'wasting time'):
						if(v>0.5):
							print(k,v)
							count9 = count9 + 1
						else:
							count9 = count9 + 1
					else:
						print(k,v)
						count9 = count9 + 1












