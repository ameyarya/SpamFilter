import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from time import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Create a dictionary of words with its frequency
train_dir = 'lingspam_public\\lemm_stop_equal\\train'
dictionary = make_Dictionary(train_dir)
# print ("Dictionary: ",dictionary)
# print ("Dictionary size: ",len(dictionary))

# Prepare feature vectors per training mail and its labels
train_labels = np.zeros(864)
train_labels[432:864] = 1
t0 = time()
# pickle_out = open("tools/train.pkl","rb")
# train_matrix = pickle.load(pickle_out)
# pickle_out.close()
train_matrix = extract_features(train_dir)
# train_matrix = np.load('train.npy')
# file = open('values.csv','r') 
# train_matrix = file.read()
# from numpy import genfromtxt
# train_matrix = genfromtxt('values.csv', delimiter=',')
# print(train_matrix.shape)
print("Extraction time for training:", round(time()-t0, 3), "s")

# Training classifiers and its variants
naive_bayes = GaussianNB()
random_forest = RandomForestClassifier(max_depth=2,random_state=0)
random_forest1 = RandomForestClassifier(max_depth=50,random_state=0)
random_forest2 = RandomForestClassifier(max_depth=100,random_state=0)
dtc = DecisionTreeClassifier(random_state=0)
svm = SVC(kernel="rbf",gamma='auto',C=10000.0)
svm1 = SVC(kernel="linear",gamma='auto',C=10000.0)
svm2 = SVC(kernel="rbf",gamma='auto',C=1.0)
kNN = KNeighborsClassifier(n_neighbors=3)
kNN1 = KNeighborsClassifier(n_neighbors=1)
kNN2 = KNeighborsClassifier(n_neighbors=10)

t0 = time()
naive_bayes.fit(train_matrix,train_labels)
print("Training time for Naive Bayes:", round(time()-t0, 3), "s")
t0 = time()
random_forest.fit(train_matrix,train_labels)
random_forest1.fit(train_matrix,train_labels)
random_forest2.fit(train_matrix,train_labels)
print("Training time for Random Forest:", round(time()-t0, 3), "s")
t0 = time()
dtc.fit(train_matrix,train_labels)
print("Training time for Decision Tree:", round(time()-t0, 3), "s")
t0 = time()
svm.fit(train_matrix,train_labels)
svm1.fit(train_matrix,train_labels)
svm2.fit(train_matrix,train_labels)
print("Training time for SVM:", round(time()-t0, 3), "s")
t0 = time()
kNN.fit(train_matrix,train_labels)
kNN1.fit(train_matrix,train_labels)
kNN2.fit(train_matrix,train_labels)
print("Training time for kNN:", round(time()-t0, 3), "s")

# Prepare feature vectors per testing mail and its labels
test_labels = np.zeros(98)
test_labels[49:98] = 1
test_dir = 'lingspam_public\\lemm_stop_equal\\test'
t0 = time()
# pickle_out = open("tools\\test.pkl","rb")
# test_matrix = pickle.load(pickle_out)
# pickle_out.close()
test_matrix = extract_features(test_dir)
# test_matrix = np.load('test.npy')
print("Extraction time for testing:", round(time()-t0, 3), "s")


# Test the unseen mails for Spam
t0 = time()
nbresult = naive_bayes.predict(test_matrix)
print("Prediction time for Naive Bayes:", round(time()-t0, 3), "s")
t0 = time()
dtresult = dtc.predict(test_matrix)
print("Prediction time for Decision Tree:", round(time()-t0, 3), "s")
t0 = time()
rfresult = random_forest.predict(test_matrix)
rfresult1 = random_forest1.predict(test_matrix)
rfresult2 = random_forest2.predict(test_matrix)
print("Prediction time for Random Forest:", round(time()-t0, 3), "s")
t0 = time()
svmresult = svm.predict(test_matrix)
svmresult1 = svm1.predict(test_matrix)
svmresult2 = svm2.predict(test_matrix)
print("Prediction time for SVM:", round(time()-t0, 3), "s")
t0 = time()
knnresult = kNN.predict(test_matrix)
knnresult1 = kNN1.predict(test_matrix)
knnresult2 = kNN2.predict(test_matrix)
print("Prediction time for kNN:", round(time()-t0, 3), "s")

# Printing the confusion matrix for all the models
print ("Confusion Matrix for Naive Bayes: \n",confusion_matrix(test_labels,nbresult))
print ("Confusion Matrix for Decision Tree: \n",confusion_matrix(test_labels,dtresult))
print ("Confusion Matrix for Random Forest: \n",confusion_matrix(test_labels,rfresult))
print ("Confusion Matrix for Random Forest 1: \n",confusion_matrix(test_labels,rfresult1))
print ("Confusion Matrix for Random Forest 2: \n",confusion_matrix(test_labels,rfresult2))
print ("Confusion Matrix for SVM : \n",confusion_matrix(test_labels,svmresult))
print ("Confusion Matrix for SVM 1: \n",confusion_matrix(test_labels,svmresult1))
print ("Confusion Matrix for SVM 2: \n",confusion_matrix(test_labels,svmresult2))
print ("Confusion Matrix for kNN : \n",confusion_matrix(test_labels,knnresult))
print ("Confusion Matrix for kNN 1: \n",confusion_matrix(test_labels,knnresult1))
print ("Confusion Matrix for kNN 2: \n",confusion_matrix(test_labels,knnresult2))

# # Printing the precision of each model
# print("Precision for Naive Bayes is:", precision_score(test_labels, nbresult)*100,"%")
# print("Precision for Decision Tree is:", precision_score(test_labels, dtresult)*100,"%")
# print("Precision for Random Forest is:", precision_score(test_labels, rfresult)*100,"%")
# print("Precision for SVM is:", precision_score(test_labels, svmresult)*100,"%")
# print("Precision for kNN is:", precision_score(test_labels, knnresult)*100,"%")

# # Printing the recall of each model
# print("Recall for Naive Bayes is:", recall_score(test_labels, nbresult)*100,"%")
# print("Recall for Decision Tree is:", recall_score(test_labels, dtresult)*100,"%")
# print("Recall for Random Forest is:", recall_score(test_labels, rfresult)*100,"%")
# print("Recall for SVM is:", recall_score(test_labels, svmresult)*100,"%")
# print("Recall for kNN is:", recall_score(test_labels, knnresult)*100,"%")

# Printing the acuracy of each model
print("Accuracy for Naive Bayes is:", accuracy_score(test_labels, nbresult)*100,"%")
print("Accuracy for Decision Tree is:", accuracy_score(test_labels, dtresult)*100,"%")
print("Accuracy for Random Forest is:", accuracy_score(test_labels, rfresult)*100,"%")
print("Accuracy for Random Forest1 is:", accuracy_score(test_labels, rfresult1)*100,"%")
print("Accuracy for Random Forest2 is:", accuracy_score(test_labels, rfresult2)*100,"%")
print("Accuracy for SVM is:", accuracy_score(test_labels, svmresult)*100,"%")
print("Accuracy for SVM1 is:", accuracy_score(test_labels, svmresult1)*100,"%")
print("Accuracy for SVM2 is:", accuracy_score(test_labels, svmresult2)*100,"%")
print("Accuracy for kNN is:", accuracy_score(test_labels, knnresult)*100,"%")
print("Accuracy for kNN1 is:", accuracy_score(test_labels, knnresult1)*100,"%")
print("Accuracy for kNN2 is:", accuracy_score(test_labels, knnresult2)*100,"%")