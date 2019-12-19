import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    
    list_to_remove = dictionary.keys()

    for item in list(dictionary):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    return dictionary
    
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix


train_dir = 'lingspam_public\\lemm_stop\\train-mails'
dictionary = make_Dictionary(train_dir)
train_matrix = extract_features(train_dir)
# np.save('train.npy',train_matrix)

# a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
# train_matrix = extract_features(train_dir)
# numpy.savetxt("foo.txt", a, delimiter=",")
# np.savetxt('values.csv', train_matrix, fmt="%d", delimiter=",")

# test_dir = 'lingspam_public\\lemm_stop\\test-mails'
# dictionary = make_Dictionary(test_dir)
# test_matrix = extract_features(test_dir)
# np.save('test.npy',test_matrix)

# file = open('values.csv','r') 
# print(file.read())

# file.close()

from numpy import genfromtxt
train_matrix1 = genfromtxt('values.csv', delimiter=',')
print(train_matrix==train_matrix1)