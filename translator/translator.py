import pandas as pd
import string
from os import path

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split

FILE_PATH = 'data/sciq/train.json'
SAVED_MODEL_PATH = 'translator/model.h1.qa_aug_20'

# build NMT model 
def build_model(in_vocab,out_vocab, in_timesteps,out_timesteps,n):   
    model = Sequential() 
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True)) 
    model.add(LSTM(n)) 
    model.add(RepeatVector(out_timesteps)) 
    model.add(LSTM(n, return_sequences=True))  
    model.add(Dense(out_vocab, activation='softmax')) 
    return model

def get_model(data):
    if (not(path.exists(SAVED_MODEL_PATH))):
      # prepare question tokenizer 
      que_tokenizer = tokenization(data['question']) 
      que_vocab_size = len(que_tokenizer.word_index) + 1 
      que_length = 50 

      # prepare answer tokenizer 
      ans_tokenizer = tokenization(data['support']) 
      ans_vocab_size = len(ans_tokenizer.word_index) + 1 
      ans_length = 50 

      # split data into train and test set 
      train, test= train_test_split(data,test_size=0.2,random_state= 12)

      # prepare training data 
      trainX = encode_sequences(ans_tokenizer, ans_length, train['support']) 
      trainY = encode_sequences(que_tokenizer, que_length, train['question'])

      # model compilation (with 512 hidden units)
      model = build_model(ans_vocab_size, que_vocab_size, ans_length, que_length, 512)
      rms = optimizers.RMSprop(lr=0.001) 
      model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

      # set checkpoint
      checkpoint = ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_loss', verbose=1, 
                      save_best_only=True, mode='min') 

      # train model 
      history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
                  epochs=30, batch_size=512, validation_split = 0.2, 
                  callbacks=[checkpoint], verbose=1)
    return load_model(SAVED_MODEL_PATH)

# function to build a tokenizer 
def tokenization(lines): 
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines) 
    return tokenizer

# encode and pad sequences 
def encode_sequences(tokenizer, length, lines):          
    # integer encode sequences          
    seq = tokenizer.texts_to_sequences(lines)          
    # pad sequences with 0 values          
    seq = pad_sequences(seq, maxlen=length, padding='post')           
    return seq

def get_word(n, tokenizer):  
    for word, index in tokenizer.word_index.items():                   
        if index == n:
            return word
    return None

def get_predicted_text(questions, preds):
    que_tokenizer = tokenization(questions)

    preds_text = [] 
    for i in preds:
        temp = []        
        for j in range(len(i)):             
            t = get_word(i[j], que_tokenizer)           
            if j > 0:
                if (t==get_word(i[j-1], que_tokenizer)) or (t== None):                       
                     temp.append('')                 
                else:                      
                     temp.append(t)             
            else:                    
                if(t == None):                                   
                     temp.append('')                    
                else:                           
                     temp.append(t)        
        preds_text.append(' '.join(temp))

    return preds_text

def translate(sentencesAndUuid):
    data = pd.read_json(FILE_PATH)
    data['question'] = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in data['question']]
    data['support'] = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in data['support']]
    ans_tokenizer = tokenization(data['support'])
    ans_length = 50

    text_encoded = encode_sequences(ans_tokenizer, ans_length, sentencesAndUuid['sentence'])
    model = get_model(data)
    preds = model.predict_classes(text_encoded.reshape((text_encoded.shape[0], text_encoded.shape[1])))
    pred_df = pd.DataFrame({'whQuestion' : get_predicted_text(data['question'], preds), 'uuid':sentencesAndUuid['uuid']})
    return pred_df
