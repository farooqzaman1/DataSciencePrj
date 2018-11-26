from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("../DataSet/train.csv")

print("done loading")

# train1  = data[:2000]


train, test = train_test_split(data, test_size=0.3)
X_train = train['question_text']
y_train = train['target']
X_test = test['question_text']
y_test = test['target']



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
vocab_filename = '../DataSet/vocab.txt'
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


###########################33
# load all training reviews
test_docs = process_docs(X_test, vocab)


# fit the tokenizer on the documents
tokenizer.fit_on_texts(test_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

###############################3


clf = BernoulliNB()
clf.fit(Xtrain, y_train)

y_pred= (clf.predict(Xtest))



sc1 = accuracy_score(y_test, y_pred)
print(sc1)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


print("tn =",tn, " fp=",fp, " fn=",fn, "tp=",tp)

p = tp/(tp+fn)
r= tp/(tp+fp)
f1=2*p*r/(p+r)
print("precission=",p,"recall=",r," F1=",f1)
