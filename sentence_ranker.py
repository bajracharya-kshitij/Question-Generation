import numpy as np
import pandas as pd
import re
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')

def rearrangeByRank(text):
    sentences = sent_tokenize(text)
    ranked_sentences = getRankedSentences(sentences)
    rearrangedByRank = []
    for sentence in ranked_sentences[:10]:
        rearrangedByRank.append(sentence[1])
    return ' '.join(rearrangedByRank)

def cleanSentences(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    return [remove_stopwords(r.split()) for r in clean_sentences]

# function to remove stopwords
def remove_stopwords(sentence):
    sen_new = " ".join([i for i in sentence if i not in stop_words])
    return sen_new


def extractWordVectors():
    word_embeddings = {}
    f = open('data/embeddings/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def getRankedSentences(sentences):
    sim_mat = getSimilarityMatrix(sentences)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

def getSimilarityMatrix(sentences):
    clean_sentences = cleanSentences(sentences)
    sentence_vectors = getSentenceVectors(clean_sentences)
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    return sim_mat

def getSentenceVectors(clean_sentences):
    word_embeddings = extractWordVectors()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

# # similarity matrix
# sim_mat = np.zeros([len(sentences), len(sentences)])

# for i in range(len(sentences)):
#     for j in range(len(sentences)):
#         if i != j:
#             sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
            
# nx_graph = nx.from_numpy_array(sim_mat)
# scores = nx.pagerank(nx_graph)

# ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# # Extract top 10 sentences as the summary
# for i in range(10):
#     print(ranked_sentences[i][1])