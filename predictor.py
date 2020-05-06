import pickle
from pathlib import Path

def pickle_exists(filename):
	file = Path('models/' + filename + '.pkl')
	if file.is_file():
		return True
	return False

def save_model(model, filename):
	pickle.dump(model, open('models/' + filename + '.pkl', 'wb'))
    
def load_model(filename):
	return pickle.load(open('models/' + filename + '.pkl', 'rb'))

def predictWords(wordsDf, df, model):
	predictor = load_model(model)

	y_pred = predictor.predict(wordsDf)
	
	labeledAnswers = []
	for i in range(len(y_pred)):
		labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i]})
	
	return labeledAnswers