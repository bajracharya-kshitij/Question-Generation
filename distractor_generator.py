import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

glove_file = 'data/embeddings/glove.6B.300d.txt'
tmp_file = 'data/embeddings/word2vec-glove.6B.300d.txt'

from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)

def generate_distractors(answer, count):
	answer = str.lower(answer)

	##Extracting closest words for the answer. 
	try:
		closestWords = model.most_similar(positive=[answer], topn=count)
	except:
		#In case the word is not in the vocabulary, or other problem not loading embeddings
		return []

	#Return count many distractors
	distractors = list(map(lambda x: x[0], closestWords))[0:count]
	
	return distractors

def addDistractors(qaPairs, count):
	for qaPair in qaPairs:
		distractors = generate_distractors(qaPair['answer'], count)
		qaPair['distractors'] = distractors
	
	return qaPairs