import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Conv1D, MaxPooling1D


cnn_model = Sequential()

cnn_model.add(Conv1D(128,3,input_shape=(train_x.shape[1:])))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Conv1D(128,3))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Conv1D(128,3))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Flatten())
cnn_model.add(Dense(32))

cnn_model.add(Dense(2, activation='softmax'))

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

cnn_history = cnn_model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(validation_x, validation_y))

#evaluation
cnn_score = cnn_model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', cnn_score[0])
print('Test accuracy:', cnn_score[1])
