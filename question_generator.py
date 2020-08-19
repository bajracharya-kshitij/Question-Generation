import spacy
import uuid
import pandas as pd

from translator import translator

nlp = spacy.load('en_core_web_sm')

def addQuestions(answers, text):
	doc = nlp(text)
	currentAnswerIndex = 0
	qaPair = []
	sentencesAndUuid = []

    #Check wheter each token is the next answer
	for sent in doc.sents:
		for token in sent:
            
			#If all the answers have been found, stop looking
			if currentAnswerIndex >= len(answers):
				break
            
			#In the case where the answer is consisted of more than one token, check the following tokens as well.
			answerDoc = nlp(answers[currentAnswerIndex]['word'])
			answerIsFound = True
            
			for j in range(len(answerDoc)):
				if token.i + j >= len(doc) or doc[token.i + j].text != answerDoc[j].text:
					answerIsFound = False
           
			#If the current token is corresponding with the answer, add it 
			if answerIsFound:
				question = blankAnswer(token.i, token.i + len(answerDoc) - 1, sent.start, sent.end, doc)
				unique_id = uuid.uuid4()

				qaPair.append({
					'question' : question,
					'answer': answers[currentAnswerIndex]['word'], 
					'prob': answers[currentAnswerIndex]['prob'],
					'uuid': unique_id})

				sentencesAndUuid.append({
					'sentence' : str(sent),
					'uuid': unique_id})          
                
				currentAnswerIndex += 1
    
	qaPairDf = pd.DataFrame(qaPair)  
	translatedDf = translator.translate(pd.DataFrame(sentencesAndUuid))
	mergedDf = pd.merge(qaPairDf, translatedDf, on='uuid')
	return mergedDf


def blankAnswer(firstTokenIndex, lastTokenIndex, sentStart, sentEnd, doc):
	leftPartStart = doc[sentStart].idx
	leftPartEnd = doc[firstTokenIndex].idx
	rightPartStart = doc[lastTokenIndex].idx + len(doc[lastTokenIndex])
	rightPartEnd = doc[sentEnd - 1].idx + len(doc[sentEnd - 1])
    
	question = doc.text[leftPartStart:leftPartEnd] + '_____' + doc.text[rightPartStart:rightPartEnd]
    
	return question

def sortAnswers(qaPairs):
	pairs = qaPairs.to_dict('records')
	orderedQaPairs = sorted(pairs, key=lambda pair: pair['prob'])
	return orderedQaPairs  