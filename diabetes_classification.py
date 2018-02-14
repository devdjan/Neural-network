from keras.models import Sequential
from keras.layers import Dense
import numpy

# random seed for reproducibility
numpy.random.seed(7)

# loading load prima indians diabetes dataset
# 5 years of medical history
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:8]
Y = dataset[:,8]

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # input layer
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # why sigmoid, i will answer later

# compile the model, adam gradient descent (optimized)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# call the function to fir to the data( training the network)
model.fit(X, Y, epochs = 1000, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



