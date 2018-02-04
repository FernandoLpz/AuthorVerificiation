import argparse
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Lambda, LSTM, Dropout, BatchNormalization, Activation

def LoadData(path_Xtrain, path_Ytrain, path_xtest, path_ytest):
# Funcin para cargar datos y separarlos para cada una de las entradas de la arquitectura siamese
	Xtrain = np.load(path_Xtrain)
	Ytrain = np.load(path_Ytrain)
	Xtest = np.load(path_xtest)
	Ytest = np.load(path_ytest)

	XtrainLeft = Xtrain[:,0:800]
	XtrainRigth = Xtrain[:,800:1600]
	XtestLeft = Xtest[:,0:800]
	XtestRigth = Xtest[:,800:1600]

	longitud = XtrainLeft.shape[1]
	dimension = XtrainLeft.shape[2]

	return XtrainLeft, XtrainRigth, Ytrain, XtestLeft, XtestRigth, Ytest, longitud, dimension

def SiameseArquitecture(longitud, dimension):
# Arquitectura Siamese

	model = Sequential()
	model.add(Conv1D(75, 12, input_shape=(longitud, dimension)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Conv1D(50, 12))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(MaxPooling1D(4))
	model.add(LSTM(64, recurrent_dropout=0.1, return_sequences=False))
	model.add(Activation('relu'))
	
	model.summary()
	return model

def euclidean_distance(vects):
# Definición de distancia ecuclidiana
	x, y = vects
	return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes	
	return (shape1[0], 1)

def compute_accuracy(predictions, labels):
# Cálculo de la exactitud
	return labels[predictions.ravel() < 0.5].mean()

parser = argparse.ArgumentParser()
parser.add_argument("-X", "--path_Xtrain", help="Path X train")
parser.add_argument("-Y", "--path_Ytrain", help="Path Y train")
parser.add_argument("-x", "--path_xtest", help="Path x test")
parser.add_argument("-y", "--path_ytest", help="Path y test")

args = parser.parse_args()
path_Xtrain = args.path_Xtrain
path_Ytrain = args.path_Ytrain
path_xtest = args.path_xtest
path_ytest = args.path_ytest

np.random.seed(9)
XtrainLeft, XtrainRigth, Ytrain, XtestLeft, XtestRigth, Ytest, longitud, dimension = LoadData(path_Xtrain,
											     path_Ytrain,
											     path_xtest,
											     path_ytest)
# Llamado de la función de la arquitectura siamese
Siamese = SiameseArquitecture(longitud, dimension)
# Declaración de cada entrada a la red neuronal
input1 = Input(shape=(longitud,dimension))
input2 = Input(shape=(longitud,dimension))
# Entrada a la red neuronal
brenchLeft = Siamese(input1)
brenchRight = Siamese(input2)
# Cálculo de la distancia euclidiana del vector resultante
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([brenchLeft, brenchRight])

rms = RMSprop()
#Inicialización del modelo
model = Model([input1,input2], distance)
model.compile(loss='mean_squared_error', optimizer=rms)

tracc, tsacc = [], []
trloss, tsloss = [], []
# Ciclo para evaluar 100 épocas
for i in range(100):
	print("->Epoch: ", i+1)
	history = model.fit([XtrainLeft, XtrainRigth], Ytrain, 
	validation_data=([XtestLeft, XtestRigth],Ytest), epochs=1, batch_size=512)
	pred = model.predict([XtrainLeft, XtrainRigth])
	tr_acc = compute_accuracy(pred, Ytrain)
	pred = model.predict([XtestLeft, XtestRigth])
	te_acc = compute_accuracy(pred, Ytest)
	print("Train acc: ", tr_acc)
	print("Test acc: ", te_acc)

# Grabado del modelo entrenado
model.save('lstm_model.h5')
