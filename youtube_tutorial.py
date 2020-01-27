

'''
so.......... install an earlier verson of tensorflow?>??


YES......... 2.0.0 works.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = data.load_data()

#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



train_images = train_images/255.0
test_images = test_images/255.0



#creating a model:
model = keras.Sequential([
	#flatten the data so it is passible to the different neurons:
	#first layer: (input layer)
	keras.layers.Flatten(input_shape = (28, 28)),
	#creating the different layers:
	#dense layers are when every node in current layer is each connected to every node in the next layer
	keras.layers.Dense(128, activation = 'relu'),
	keras.layers.Dense(10, activation ='softmax')
	])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#epoch is giving the same images for a given amount of times
#epochs are used so to get a more accurate result given by different order of picutres
model.fit(train_images, train_labels, epochs = 5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('tested acc', test_acc)

prediction = model.predict(test_images[7])

#for loop to show the actual image to show testing image:
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel('actual: '+ class_names[test_labels[i]])
    plt.title('Prediction: '+ class_names[np.argmax(prediction[i])])
    plt.show()
    
print(class_names[np.argmax(prediction[0])])















