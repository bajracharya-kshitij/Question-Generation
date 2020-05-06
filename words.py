import spacy

nlp = spacy.load('en_core_web_sm')

def addWordsForNewParagraph(newWords, text):
    doc = nlp(text)

    neStarts = getNEStartIndexes(doc)
    sentenceStarts = getSentenceStartIndexes(doc)
    
    i = 0
    
    while(i<len(doc)):
        #If the token is a start of a Named Entity, add it and push to index to end of the NE
        if (i in neStarts):
            word = neStarts[i]
            currentSentence = getSentenceForWordPosition(word.start, sentenceStarts)
            wordLength = word.end - word.start
            shape = ''
            for wordIndex in range(word.start, word.end):
                shape += (' ' + doc[wordIndex].shape_)
                
            newWords.append([word.text, 0, 0, currentSentence, wordLength, word.label_, None, None, None, shape])
            
            i = neStarts[i].end - 1
        else:
            #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
            if (doc[i].is_stop == False and doc[i].is_alpha == True):
                word = doc[i]
                currentSentence = getSentenceForWordPosition(i, sentenceStarts)
                wordLength = 1
                newWords.append([word.text, 0, 0, currentSentence, wordLength, None, word.pos_, word.tag_, word.dep_, word.shape_])
        i += 1

def addWordsForParagraph(df, newWords, titleId, paragraphId):
    text = df['data'][titleId]['paragraphs'][paragraphId]['context']
    qas = df['data'][titleId]['paragraphs'][paragraphId]['qas']
    
    doc = nlp(text)
    
    answers = extractAnswers(qas, doc)
    neStarts = getNEStartIndexes(doc)
    sentenceStarts = getSentenceStartIndexes(doc)
    
    i = 0
    
    while(i<len(doc)):
        #If the token is a start of a Named Entity, add it and push to index to end of the NE
        if (i in neStarts):
            word = neStarts[i]
            currentSentence = getSentenceForWordPosition(word.start, sentenceStarts)
            wordLength = word.end - word.start
            shape = ''
            for wordIndex in range(word.start, word.end):
                shape += (' ' + doc[wordIndex].shape_)
                
            newWords.append([word.text, tokenIsAnswer(word.text, currentSentence, answers), titleId, paragraphId, currentSentence, wordLength, word.label_, None, None, None, shape])
            
            i = neStarts[i].end - 1
        else:
            #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
            if (doc[i].is_stop == False and doc[i].is_alpha == True):
                word = doc[i]
                currentSentence = getSentenceForWordPosition(i, sentenceStarts)
                wordLength = 1
                newWords.append([word.text, tokenIsAnswer(word.text, currentSentence, answers), titleId, paragraphId, currentSentence, wordLength, None, word.pos_, word.tag_, word.dep_, word.shape_])
        i += 1

def getSentenceForWordPosition(wordPos, sentenceStarts):
    for i in range(1, len(sentenceStarts)):
        if (wordPos < sentenceStarts[i]):
            return i - 1

def getNEStartIndexes(doc):
    neStarts = {}
    for ne in doc.ents:
        neStarts[ne.start] = ne
        
    return neStarts

def getSentenceStartIndexes(doc):
    sentenceStarts = []
    
    for sentence in doc.sents:
        sentenceStarts.append(sentence[0].i)
    
    return sentenceStarts

def extractAnswers(qas, doc):
    answers = []
    
    sentenceStart = 0
    sentenceId = 0
    
    for sentence in doc.sents:
        sentenceLength = len(sentence.text)
        
        for answer in qas:
            answerStart = answer['answers'][0]['answer_start']
            if (answerStart >= sentenceStart and answerStart < (sentenceStart + sentenceLength)):
                answers.append({'sentenceId': sentenceId, 'text': answer['answers'][0]['text']})
                
        sentenceStart += sentenceLength
        sentenceId += 1
                    
    return answers

def tokenIsAnswer(token, sentenceId, answers):
    for i in range(len(answers)):
        if (answers[i]['sentenceId'] == sentenceId):
            if (answers[i]['text'] == token):
                return True
    return False