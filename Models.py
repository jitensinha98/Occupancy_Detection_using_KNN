import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

def prep_data():

	# Reading divided datasets as pandas dataframe
	data_1 = pd.read_csv('Dataset/datatraining.csv')
	data_2 = pd.read_csv('Dataset/datatest.csv')
	data_3 = pd.read_csv('Dataset/datatest2.csv')

	frames = [data_1,data_2,data_3]
	
	# concatination three datasets into one
	dataset = pd.concat(frames)
	
	print("Dataset Dimension :")	
	print(dataset.shape)
	print(" ")

	print("Dataset Sample :")
	print(dataset.head(10))
	print(" ")

	# removing date column
	dataset = dataset.drop(['date'],axis=1)

	print("Dataset Sample after modifying :")
	print(dataset.head(10))
	print(" ")
	
	# x contains the feature matrix
	# y_labels contains the labels
	x = dataset[dataset.columns[0:(dataset.shape[1]-1)]].values
	y = dataset[dataset.columns[dataset.shape[1]-1]]

	# one-hot encoding y
	y = pd.get_dummies(y)

	# converting y to numpy array
	y = np.array(y)

	# shuffling dataset
	x,y = shuffle(x,y,random_state = 1)

	# splitting dataset into training and testing data
	train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.20)

	return train_x,test_x,train_y,test_y

def KNN_model():
	print("----------------------KNN CLASSIFIER---------------------------")
	model_knn = KNeighborsClassifier(n_neighbors=n_classes)

	print("Training KNN model...")
	model_knn.fit(train_x,train_y)
	pred_y = model_knn.predict(test_x)
	print("Training Done.")
	print(" ")

	acc = metrics.accuracy_score(test_y,pred_y)
	print("Test Accuracy = ",acc*100)
	print(" ")

	print("Saving KNN Model..")
	filename = 'KNN_Classifier_model.sav'
	pickle.dump(model_knn, open(filename, 'wb'))
	print("Model Saved.")
	print("----------------------------------------------------------------")

def MLP_model():
	print("----------------------MLP CLASSIFIER---------------------------")
	model_mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 50), max_iter=70)

	print("Training MLP model...")
	model_mlp.fit(train_x,train_y)
	pred_y = model_mlp.predict(test_x)
	print("Training Done.")
	print(" ")

	acc = metrics.accuracy_score(test_y,pred_y)
	print("Test Accuracy = ",acc*100)
	print(" ")

	print("Saving MLP Model..")
	filename = 'MLP_Classifier_model.sav'
	pickle.dump(model_mlp, open(filename, 'wb'))
	print("Model Saved.")
	print("----------------------------------------------------------------")

def RF_model():
	print("----------------------RANDOM FOREST CLASSIFIER---------------------------")
	model_rf = RandomForestClassifier(n_estimators = 100)

	print("Training RANDOM FOREST model...")
	model_rf.fit(train_x,train_y)
	pred_y = model_rf.predict(test_x)
	print("Training Done.")
	print(" ")

	acc = metrics.accuracy_score(test_y,pred_y)
	print("Test Accuracy = ",acc*100)
	print(" ")

	print("Saving RANDOM FOREST Model..")
	filename = 'RANDOM_FOREEST_Classifier_model.sav'
	pickle.dump(model_rf, open(filename, 'wb'))
	print("Model Saved.")
	print("----------------------------------------------------------------")



n_classes = 2
train_x,test_x,train_y,test_y = prep_data()
KNN_model()
MLP_model()
RF_model()
