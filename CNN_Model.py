from Libraries import *
from Training_Data import *
from Validation_Data import *
# create_training_data()
# create_val_data()
model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3), activation = 'relu',input_shape = x_train.shape[1:])) # OR input_shape = (img_size,img_size,1)
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(.2))

model.add(Conv2D(128,kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(.2))

model.add(Conv2D(256,kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1,activation = 'sigmoid'))

