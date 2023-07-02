import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

data_size = 2000
train_data = np.random.randint(100,size=(data_size,2))

train_ans =(train_data[:,0]+train_data[:,1])
print(np.shape(train_ans))

model=keras.Sequential()
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(1, activation='elu'))
model.compile(loss='mse', optimize=tf.keras.optimizers.Adam(0.88), metrics=['Accuracy'])
model.fit(train_data,train_ans,batch_size=1,epochs=20)
z=np.array([10.1,20.3]).reshape(1,2)
z=np.array([100,200]).reshape(1,2)
q=model.predict(z, batch_size=1)
print(q)