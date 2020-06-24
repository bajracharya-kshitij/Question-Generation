import words as wd
import preprocessing as pp

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

squadPath = 'data/squad-v1/'
datasetFile = 'data/squad-v1/squad.csv'

def getProcessedSquad():
	df = getDf(squadPath)

	file = Path(datasetFile)

	if file.is_file():
		print("File exists, loading from file")
		squad = pd.read_csv(datasetFile)
	else:
		print ("File doesn't exist. Creating...")
		
		words = []
		wordColumns = ['text', 'isAnswer', 'titleId', 'paragraphId', 'sentenceId', 'wordCount', 'NER', 'POS', 'TAG', 'DEP', 'shape']
		titlesCount = len(df['data'])
	    
		count = 0
		for titleId in range(titlesCount):
			paragraphsCount = len(df['data'][titleId]['paragraphs'])
			for paragraphId in range(paragraphsCount):
				wd.addWordsForParagraph(df, words, titleId, paragraphId)
				count += 1
				if (count%1000 == 0):
					print(count)
		
		squad = pd.DataFrame(words, columns=wordColumns)
		squad.to_csv(datasetFile, index=False)
		print('100% done and written to file')

	return squad

def getDf(path):
	train = pd.read_json(squadPath + 'train-v1.1.json', orient='column')
	dev = pd.read_json(squadPath + 'dev-v1.1.json', orient='column')

	return pd.concat([train, dev], ignore_index=True)

def start(modelFile, classifier):
	df = getProcessedSquad()
	df = pp.encodeAndDropColumns(df)

	x_data = df.drop(labels=['isAnswer'], axis=1)
	y_data = df['isAnswer']
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=4)

	if pickle_exists(modelFile):
		print('\nPickle already exists. Loading from file ' + modelFile + '.pkl')
		model = load_model(modelFile)
	else:
		print('\nNo pickle found. Training model and saving to file ' + modelFile + '.pkl')

		model = classifier.fit(x_train, y_train)
		save_model(model, modelFile)

	y_pred = model.predict(x_test)
	correctCount = (y_test == y_pred).sum()
	print('Correctly guessed:', '{:.2f}%'.format((correctCount / len(y_test)) * 100))
	print('Accuracy score: ', accuracy_score(y_test, y_pred))

def pickle_exists(filename):
    file = Path('models/' + filename + '.pkl')
    if file.is_file():
        return True
    return False

def save_model(model, filename):
    pickle.dump(model, open('models/' + filename + '.pkl', 'wb'))
    
def load_model(filename):
    return pickle.load(open('models/' + filename + '.pkl', 'rb'))

start('gaussian_naive_bayes', GaussianNB())
start('logistic_regression', LogisticRegression())
