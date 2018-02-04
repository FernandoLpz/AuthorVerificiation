# Author verificiation using a siamese arquitecture 
This repository shows up a siamese arquitectue proposed to solve the problem of author verification particularly the problem about given a pair of documents decide if both are from the same author or not based on their writting style. The siamese arquitecture is composed by an assemble of two convolutional layers and a LSTM recurrent neurnal net followed by a euclidean distance.

The framework implemented was Keras 2.0 on Python 3.5.2.

# Introduction
The model proposed is based on the idea that it can learn a function that may decide that given a pair of documents if both are from the same author or not based on their writting style.
<p align="center">
  <img src="https://github.com/FernandoLpz/AuthorVerificiation/blob/master/images/verification.png" width="350"/>
</p>

# The model
The model receives two inputs wich are a pair of sequences of character embeddings that represents a document from a author given each one. Each sequence pass through a first convolutional layer wich extract local features from fragments of the sequence, next the second convolutional layer extracts features from the features extracted from the first convolutional layer, next the maxpooling layer extracts the most important feature wich describes the author's style, after that these features are passed through a LSTM wich learns the sign or the author's style.
