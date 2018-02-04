# Author verificiation using a siamese arquitecture 
This repository shows up a siamese arquitectue proposed to solve the problem of author verification particularly the problem about given a pair of documents decide if both are from the same author or not based on their writting style. The siamese arquitecture is composed by an assemble of two convolutional layers and a LSTM recurrent neurnal net followed by a euclidean distance.

The framework implemented was Keras 2.0 on Python 3.5.2.

# Introduction
The model proposed is based on the idea that it can learn a function that may decide that given a pair of documents if both are from the same author or not based on their writting style.

![alt text](AuthorVerificiation/images/verification.png)

The model receives two inputs wich are a pair of sequences of character embeddings that represents a document from a author given each one. 
