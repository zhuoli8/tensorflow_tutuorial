import tensorflow as td
from tensorflow import keras
import numpy as np

#text classification

#loading data:
data = keras.datasets.imdb


#splitting dataset into training and test:
(train_data, train_labels),(test_data, test_labels) = data.load_data(num_words = 88000) #num_words is set to 10000 to only catch the 10000 most reoccouring words

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
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)

#print('train data length: ',len(train_data), 'test data length', len(test_data))

#returns the value(which is now a word value)
def decode_review(text):
    return" ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[0]))


#model in now done
    
model = keras.Sequential()

#creating layers:
model.add(keras.layers.Embedding(88000, 16)) #
model.add(keras.layers.GlobalAveragePooling1D()) #
model.add(keras.layers.Dense(16, activation = 'relu')) #
model.add(keras.layers.Dense(1, activation = 'sigmoid')) #sigmoid scales values to a range of 0, 1

'''
1:29:06
'''
#EXPLANATION OF THE NETWORK AND ITS LAYERS:

'''
1:43:00 last left off
'''


model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #loss function calculates the difference between the sigmoid output and 0


#splitting training data into validation data

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

#fitting model:

fitModel = model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)

results = model.evaluate(test_data, test_labels)

print('this is the result: ', results)


#1:49:05


#test results:

#test_review = test_data[0]
#predict = model.predict([test_review])
#print('Review: ')
#print(decode_review(test_review))
#print('Prediction: ' + str(predict[0]))
#print('Actual: ' + str(test_labels[0]))
#print(results)

#1:53:12

#vocab size is at 880000

def review_encode(s):
    encoded = [1]
    
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
            
    return encoded
#saving the model:
#model.save('model.h5')

#calling the model:
model = keras.models.load_model("model.h5")

with open('movie_review.txt') as fin:
    for line in fin.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index["<PAD>"], padding = "post", maxlen = 250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])







