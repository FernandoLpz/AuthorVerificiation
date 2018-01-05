import numpy as np
from keras import backend as K
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Bidirectional, Conv1D, MaxPooling1D, Dense, Lambda, LSTM, Dropout, BatchNormalization, Activation
from keras.regularizers import l2

def LoadData():
	Xtrain = np.load('../../datos/charEmbedding400x300/XtrainCharEmbedding.npy')
	Ytrain = np.load('../../datos/charEmbedding400x300/Ytran.npy')
	Xtest = np.load('../../datos/charEmbedding400x300/XtestCharEmbedding.npy')
	Ytest = np.load('../../datos/charEmbedding400x300/Ytest.npy')

	XtrainLeft = Xtrain[:,0:400]
	XtrainRigth = Xtrain[:,400:800]
	XtestLeft = Xtest[:,0:400]
	XtestRigth = Xtest[:,400:800]

	longitud = XtrainLeft.shape[1]
	dimension = XtrainLeft.shape[2]

	return XtrainLeft, XtrainRigth, Ytrain, XtestLeft, XtestRigth, Ytest, longitud, dimension

def NeuralNet(longitud, dimension):
	model = Sequential()
	model.add(Conv1D(75, 12, input_shape=(longitud, dimension)))
	model.add(Activation('relu'))
	model.add(Dropout(0.01))
	model.add(BatchNormalization())
	model.add(Conv1D(50, 12))
	model.add(Activation('relu'))
	model.add(Dropout(0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(4))
	model.add(LSTM(64, recurrent_dropout=0.1, return_sequences=False))
	model.add(Activation('relu'))
	model.summary()
	return model

def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes	
	return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
	return labels[predictions.ravel() < 0.5].mean()

np.random.seed(9)
XtrainLeft, XtrainRigth, Ytrain, XtestLeft, XtestRigth, Ytest, longitud, dimension = LoadData()

Siamese = NeuralNet(longitud, dimension)
input1 = Input(shape=(longitud,dimension))
input2 = Input(shape=(longitud,dimension))

brenchLeft = Siamese(input1)
brenchRight = Siamese(input2)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([brenchLeft, brenchRight])

rms = RMSprop()
model = Model([input1,input2], distance)
model.compile(loss=contrastive_loss, optimizer=rms)

tracc, tsacc = [], []
trloss, tsloss = [], []
for i in range(300):
	print("---------- Epoch: ", i+1)
	history = model.fit([XtrainLeft, XtrainRigth], Ytrain, 
	validation_data=([XtestLeft, XtestRigth],Ytest), epochs=1, batch_size=512)
	pred = model.predict([XtrainLeft, XtrainRigth])
	tr_acc = compute_accuracy(pred, Ytrain)
	pred = model.predict([XtestLeft, XtestRigth])
	te_acc = compute_accuracy(pred, Ytest)
	print("Train acc: ", tr_acc)
	print("Test acc: ", te_acc)

	tracc.append(tr_acc)
	tsacc.append(te_acc)
	trloss.append(history.history['loss'])
	tsloss.append(history.history['val_loss'])

np.save('TrainAcc.npy', tracc)
np.save('TestAcc.npy', tsacc)
np.save('TrainLoss.npy',trloss)
np.save('TestLoss.npy',tsloss)
# summarize history for loss
model.save('lstm_model.h5')

xrango = np.arange(0, 210, 10)
yrango = np.arange(0.4, 1, 0.05)
y_rango = np.arange(0.4, 0, -0.05)

fig = plt.figure()
plt.plot(trloss, color='green')
plt.plot(tsloss, color='blue')
plt.title('Pérdida en el modelo')
plt.ylabel('Error')
plt.xlabel('Epocas')
leg = plt.legend(['Entrenamiento', 'Prueba'], loc='upper right')
leg_lines = leg.get_lines()
plt.setp(leg_lines, linewidth=3)
plt.xticks(xrango)
plt.yticks(y_rango)
plt.grid()
fig.savefig('error.png')

fig1 = plt.figure()
plt.plot(tracc, color='teal', linewidth=0.4)
plt.plot(tsacc, color='tomato', linewidth=0.4)
plt.title('Efectividad en el modelo')
plt.ylabel('Efectividad')
plt.xlabel('Epocas')
leg = plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
leg_lines = leg.get_lines()
plt.setp(leg_lines, linewidth=3)
plt.xticks(xrango)
plt.yticks(yrango)
plt.grid()
fig1.savefig('efectividad.png')
