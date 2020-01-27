import tensorflow as td
from tensorflow import keras
import numpy as np

#text classification

#loading data:
data = keras.datasets.imdb


#splitting dataset into training and test:
(train_data, train_labels),(test_data, test_labels) = data.load_data(num_words = 10000) #num_words is set to 10000 to only catch the 10000 most reoccouring words

'''
#print(train_data[0])
^ the words in the data set are now represented by a number per word
'''

#mapping for words: (dictionary of words and values)
word_index = data.get_word_index()

#print('this is the word index: ',word_index)

#adding 3 to all the values so we can add own values for this following things:
word_index = {k:(v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0 #makes every review the same length
word_index['<START>'] = 1 
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

#swap key(word) with value(number): *rember that the dataset in now in numeric form*
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#padding of the reviews
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen =250)
test_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen =250)

#print('train data length: ',len(train_data), 'test data length', len(test_data))

#returns the value(which is now a word value)
def decode_review(text):
    return" ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[0]))


#model in now done
    
model = keras.Sequential()

#creating layers:
model.add(keras.layers.embedding(1000, 16)) #
model.add(keras.layers.GobalAveragePooling1D()) #
model.add(keras.layers.Dense(16, activation = 'relu')) #
model.add(keras.layers.Dense(1, activation = 'sigmoid')) #sigmoid scales values to a range of 0, 1

'''
1:29:06
'''
#EXPLANATION OF THE NETWORK AND ITS LAYERS:
'''

'''






























