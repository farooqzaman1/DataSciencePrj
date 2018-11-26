import numpy as np
from  keras.utils import  to_categorical
import pandas as pd
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix




data = pd.read_csv("./DataSet/train.csv")
# test = pd.read_csv("./Dataset/train.csv")
print("done loading")
# length = len(train1['question_text'])
train1  = data[:8000]
# test = train1[2001:4000]
# train.to_csv("./DataSet/SubSetTrain.csv", sep=',', encoding='utf-8')


train, test = train_test_split(train1, test_size=0.3)

X_train = train['question_text']
y_train = train['target']

X_test = test['question_text']
y_test = test['target']

# X_test = test['question_text']
# data['qid']

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r', encoding="utf-8")
	text = file.read()
	file.close()
	return text


# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens


# load all docs in a directory
def process_docs(docs, vocab,):
    documents = list()
    for d in docs:
        tokens = clean_doc(d, vocab)
        documents.append(tokens)
    return documents

# load the vocabulary
vocab_filename = './DataSet/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
train_docs = process_docs(X_train, vocab)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')



# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=6, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))
print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network



y= to_categorical(y_train, num_classes=2)


model.fit(Xtrain, y, epochs=10, verbose=2)

# serialize model to JSON
model_json = model.to_json()
with open("./Model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./Model/model.h5")
print("Saved model to disk")
# load all test reviews

test_docs = process_docs(X_test, vocab)
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# evaluate
print("Evaluating the model....")
yt = to_categorical(y_test)
loss, acc = model.evaluate(Xtest, yt, verbose=0)
print('Test Accuracy: %f' % (acc*100))

y_pred =model.predict(Xtest)

pred=[np.argmax(a)  for a in y_pred ]
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("tn =",tn, " fp=",fp, " fn=",fn, "tp=",tp)

p = tp/(tp+fn)
r= tp/(tp+fp)
f1=(2*p*r)/(p+r)
print("Accuracy=",(tp+tn/tp+tn+fp+fn),"precission=",p,"recall=",r," F1=",f1)