import random
import numpy as np

import matplotlib.pyplot as plt
from functools import reduce

from skimage.io import imread
from skimage.transform import rescale
from skimage.feature import hog

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

def getPie_andMult(s):
    xx = s[s.find('_')+1:]
    return int(s[:s.find('_')]), int(xx[:xx.find('_')])

def getTrainingData_multiplier():
    
    training_picture_files = []

    with open('./dart_images/dart_images.txt','r') as f1:
        training_picture_files += f1.readlines()
        
    with open('./dart_images/dart_images_2.txt','r') as f1:
        training_picture_files += f1.readlines()
    
    data = []
    clazz_pie = []
    clazz_mult = []
    
    random.shuffle(training_picture_files)
    
    for t in training_picture_files:
        data += [reduce(lambda z, y :z + y, imread('./dart_images/' + t.strip() + '.jpg', as_gray=True)) ]
        pie, mult = getPie_andMult(t.strip())
        clazz_pie += [pie]
        clazz_mult += [mult]
    return data, clazz_pie, clazz_mult

def getTestingData_multiplier():
    
    testing_picture_files = []

    with open('./dart_test_images/dart_test_images.txt','r') as f1:
        testing_picture_files += f1.readlines()
        
    
    data = []
    clazz_pie = []
    clazz_mult = []
    
    random.shuffle(testing_picture_files)
    
    for t in testing_picture_files:
        data += [reduce(lambda z, y :z + y, imread('./dart_images/' + t.strip() + '.jpg', as_gray=True)) ]
        pie, mult = getPie_andMult(t.strip())
        clazz_pie += [pie]
        clazz_mult += [mult]
    return data, clazz_pie, clazz_mult

def getMultiplierModel(X_train, y_train, X_test, y_test):

    training_data = np.array(X_train)
    training_class = np.array(y_train)
    testing_data = np.array(X_test)
    testing_class = np.array(y_test)

    scaler = StandardScaler()
    scaler.fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)

    pca = PCA(0.99)
    pca.fit(training_data)
    
    training_data = pca.transform(training_data)
    testing_data = pca.transform(testing_data)

    sgd_clf = RandomForestClassifier()
    sgd_clf.fit(training_data, training_class)


    print('Multiplier Accuracy')
    y_pred_train = sgd_clf.predict(training_data)
    y_pred_test = sgd_clf.predict(testing_data)
    print('Training - Percentage correct: ', 100*np.sum(y_pred_train == training_class)/len(training_class))
    print('Testing - Percentage correct: ', 100*np.sum(y_pred_test == testing_class)/len(testing_data))


    return sgd_clf

def getPieModel(X_train, y_train, X_test, y_test, ms = 0.25, mf = 0.9):

    training_data = np.array(X_train)
    training_class = np.array(y_train)
    testing_data = np.array(X_test)
    testing_class = np.array(y_test)

    scaler = StandardScaler()
    scaler.fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)

    pca = PCA(0.99)
    pca.fit(training_data)
    
    training_data = pca.transform(training_data)
    testing_data = pca.transform(testing_data)

    sgd_clf = BaggingClassifier(RandomForestClassifier(n_estimators=250), max_samples=ms, max_features=mf)
    sgd_clf.fit(training_data, training_class)

    print('Pie Accuracy')
    y_pred_train = sgd_clf.predict(training_data)
    y_pred_test = sgd_clf.predict(testing_data)
    print('Training - Percentage correct: ', 100*np.sum(y_pred_train == training_class)/len(training_class))
    print('Testing - Percentage correct: ', 100*np.sum(y_pred_test == testing_class)/len(testing_data))


    return 100*np.sum(y_pred_test == testing_class)/len(testing_data)

def getCounts(l):
    c = []
    for x in set(l):
        c += [l.count(x)]
    return c


## standard 
training_data, p_training_class, m_training_class = getTrainingData_multiplier()
testing_data, p_testing_class, m_testing_class = getTestingData_multiplier()

mult_model = getMultiplierModel(training_data, m_training_class, testing_data, m_testing_class )
pie_model = getPieModel(training_data, p_training_class, testing_data, p_testing_class, ms=ss, mf=ff)
