# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
dataset.count
y=dataset.iloc[:,1].values
import re
import nltk as nl
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nl.download('stopwords')
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=dataset['Review'][i]
    review=re.sub('[^a-zA-z]',' ',review)
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set( stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)
pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,pred)