import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

glove_file = 'data/embeddings/glove.6B.300d.txt'
tmp_file = 'data/embeddings/word2vec-glove.6B.300d.txt'

from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)

from nltk.stem import PorterStemmer

def generate_distractors(answer, count):
	answer = str.lower(answer)

	##Extracting closest words for the answer. 
	try:
		closestWords = model.most_similar(positive=[answer], topn=count)
	except:
		#In case the word is not in the vocabulary, or other problem not loading embeddings
		return []

	#Return count many distractors
	distractors = list(map(lambda x: x[0], closestWords))
	distractors = normalizeDistractors(answer, distractors, count)
	return distractors

def addDistractors(qaPairs, count):
	filteredQAPairs = []
	for qaPair in qaPairs:
		distractors = generate_distractors(qaPair['answer'], count)
		if (len(distractors)>=count):
			qaPair['distractors'] = distractors
			filteredQAPairs.append(qaPair)

	return filteredQAPairs

def normalizeDistractors(answer, distractors, count):
	porter = PorterStemmer()
	filteredDistractors = []
	normalizedDistractors = []

	answerStem = porter.stem(answer)
	for distractor in distractors:
		distractorStem = porter.stem(distractor)
		if ((distractorStem == answerStem) or (distractorStem in normalizedDistractors)):
			continue
		normalizedDistractors.append(distractorStem)
		filteredDistractors.append(distractor)
	return filteredDistractors[0:count]