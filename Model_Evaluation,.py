from CNN_Model import *
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,batch_size=4, epochs = 10, verbose=1,validation_data = (x_val,y_val))

score  = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epoch_list = range(1,len(hist.history['accuracy']) + 1) 
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy']) 
plt.legend(('Training Accuracy','Validation Accuracy'))
plt.show()

model.summary()