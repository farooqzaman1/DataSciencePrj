from collections import Counter
from math import log
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import pandas as pd
from random import shuffle

class DataLoader(object):
    def __init__(DLobject):
        print()

    def tokenize(DCobject, text):
        # print(set([w.lower() for w in text.split(" ")]))
        words = set(word_tokenize(str(text)))
        return words


    def get_labeled_data(DCobject,obj2):
        examples = []
        labels = []
        df = pd.read_csv("../Dataset/SubSetTrain.csv", encoding='utf-8')
        counter=0
        for i ,j in zip(df["question_text"], df["target"]):
            tokens = DCobject.tokenize(i)
            tokenss = obj2.removalStopWords(tokens)
            removedFunctuation = obj2.removeFunctuation(tokenss)
            stemmed = obj2.stemming(removedFunctuation)
            examples.append(stemmed)
            labels.append(j)
        return examples, labels

##################################################

class DataCleaning(object):
    def __init__(Fextract):
        print()

    def removalStopWords(Fextract, words):
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in words if not w in stop_words]
        return tokens

    def removeFunctuation(self , tokens):
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        return words
    def tokenize(DCobject, text):
        # print(set([w.lower() for w in text.split(" ")]))
        words = set(word_tokenize(str(text)))
        return words

    def stemming(self, words):
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        return stemmed

################### Model ##########################################
class model(object):
    def __init__(NBModelObject):
        NBModelObject.logPriors = {}
        NBModelObject.vocabulary = {}
        NBModelObject.classesAndVectors = []

    def train(NBModelObject, train_dcs, train_lbls):
        label_counts = Counter(train_lbls)
        N = float(sum(label_counts.values()))
        NBModelObject.logPriors = {k: log(v / N) for k, v in label_counts.items()}
        vectors = []
        for d  in train_dcs:
            for term in d:
                if term not in NBModelObject.vocabulary:
                    NBModelObject.vocabulary[term] = len(NBModelObject.vocabulary)

        for d, l in zip(train_dcs, train_lbls):
             arr = np.zeros(shape=(1, len(NBModelObject.vocabulary)))
             for term in d:
                if (term in NBModelObject.vocabulary):
                    id = NBModelObject.vocabulary[term]
                    arr[0][id] = 1
             vectors.append(tuple((arr,l)))

        for label in set(train_lbls):
            summm = np.zeros(shape=(1, len(NBModelObject.vocabulary)))
            for v,l in vectors:
                if(l==label):
                   summm= np.add(summm,v)

            for word, i in zip(NBModelObject.vocabulary, range(len(NBModelObject.vocabulary))):
             N = label_counts[label]
            NBModelObject.classesAndVectors.append(tuple((label, (summm+1./N+2.))))

    def predict(NBModelObject,clean, text):
        words  = clean.tokenize(text)
        remoovedFunct = clean.removeFunctuation(words)
        stopwrdsREmoved  = clean.removalStopWords(remoovedFunct)
        stemmed = clean.stemming(stopwrdsREmoved)
        arr = np.zeros(shape=(1, len(NBModelObject.vocabulary)))
        for term in stemmed:
            if (term in NBModelObject.vocabulary):
                idd = NBModelObject.vocabulary[term]
                arr[0][idd] = 1

        pred_class = None
        max_ = float("-inf")
        #  MAP estimation
        for l, v in NBModelObject.classesAndVectors:
            log_sum = NBModelObject.logPriors[l]
            for i,j in zip(v[0] ,range(len(NBModelObject.vocabulary))):
                if arr[0][j]==1.0:
                   prob1 = v[0][j]
                   log_sum+=prob1
                else:
                    prob2 = 1. - log((v[0][j]),2)
                    log_sum+= prob2

            if log_sum > max_:
                max_ = log_sum
                pred_class = l
        return pred_class

    def saveModel(NBModelObject):
        modelFile = open("model.txt", "w", encoding='utf-8')
        for l,v in NBModelObject.classesAndVectors:
            line = str(l)+":"
            for val in v[0]:
                line+= str(val)+","
            line = str(line).lstrip(",")
            line+="\n"
            modelFile.write(line)
        modelFile.close()
        vocabularyFile = open("vocabulary.txt", "w", encoding='utf-8')
        for key , value in NBModelObject.vocabulary.items():
            line = str(key)+":"+str(value)+"\n"
            vocabularyFile.write(line)
        vocabularyFile.close()

        priorProbFile = open("priorProbFile.txt", "w", encoding='utf-8')
        for key, value in NBModelObject.logPriors.items():
            line = str(key) + ":" + str(value) + "\n"
            priorProbFile.write(line)
        priorProbFile.close()

    def loadModel(NBModelObject):

        modelFile = open("model.txt", "r", encoding='utf-8')
        NBModelObject.vocabulary.clear()
        NBModelObject.classesAndVectors.clear()
        NBModelObject.logPriors.clear()
        labels = []
        probs = []
        for line in modelFile:
            li  =line.split("\n")
            labelandVec = li[0].split(":")
            labels.append(labelandVec[0])
            lst = []
            lst2 = []
            for ll in labelandVec[1].split(","):
                if ll!="":
                    lst.append(float(ll))
            lst2.append(lst)
            probs.append(lst2)
        for l, v in zip(labels, probs):
            NBModelObject.classesAndVectors.append(tuple((l,v)))
        vocabularyFile = open("vocabulary.txt", "r", encoding='utf-8')
        for line in vocabularyFile:
            li = line.split("\n")
            keyandValue = li[0].split(":")
            NBModelObject.vocabulary[keyandValue[0]] = int(keyandValue[1])

        priorprobFile = open("priorProbFile.txt", "r", encoding='utf-8')
        for line in priorprobFile:
            li = line.split("\n")
            keyandValue = li[0].split(":")
            if keyandValue[1]!="":
             NBModelObject.logPriors[keyandValue[0]] = float(keyandValue[1])
####  END of Model....



dCleaning  = DataCleaning()
dLoader = DataLoader()

train_docs, train_labels = dLoader.get_labeled_data(dCleaning)

objectt  =  model()

listtt = []
for i, j in zip(train_docs, train_labels):
    listtt.append(tuple((i,j)))

totalAcc=0
for i in range(10):
    shuffle(listtt)
    totalDoc = len(train_docs)
    test = listtt[:int(0.4*totalDoc)]
    train = listtt[int(0.4*totalDoc)+1:]
    trainD = []
    trainL = []
    for p in listtt:
        trainD.append(p[0])
        trainL.append(p[1])
    objectt.train(trainD, trainL)
    cont = 0
    for p in test:
        d = p[0]
        a = p[1]
        A = objectt.predict(dCleaning, d)
        if a==A:
            cont+=1
    acc = cont*100.0/len(test)
    print("acc = ",i+1,": ",round((cont*100.0/len(test)),2))
    totalAcc+=acc

print("Avg Acc = ",round(totalAcc/10,2))

