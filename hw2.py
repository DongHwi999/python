import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

data_size = 10000

# train_data = np.random.randint(100, size=(data_size, 2))-50+np.random.randint(100,size=(data_size,2))/100 #덧셈과 뺄셈 훈련데이터
train_data = np.random.randint(100, size=(data_size, 2))+1e-6 #곱셈과 나눗셈 훈련데이터

# 뒤에 +를 통해 소수값을 넣음
train_ans = (train_data[:, 0] + train_data[:, 1])  # 덧셈
train_ans2 = (train_data[:, 0] - train_data[:, 1])  # 뺄셈
train_ans3 = (train_data[:, 0] * train_data[:, 1]) # 곱셈
train_ans3 = np.log(train_ans3)
train_ans4 = (train_data[:, 0] / train_data[:, 1]) # 나눗셈
train_ans4 = np.log(train_ans4)



model = keras.Sequential()

model.add(keras.layers.Dense(10, input_dim=2, activation='linear'))
model.add(keras.layers.Dense(20, activation='linear'))
model.add(keras.layers.Dense(5, activation='linear'))
model.add(keras.layers.Dense(1, activation='linear'))
# 이건 덧셈과 뺄셈용

#model.add(keras.layers.Dense(10, input_dim=2, activation='relu'))
#model.add(keras.layers.Dense(20, activation='relu'))
#model.add(keras.layers.Dense(5, activation='linear'))
#model.add(keras.layers.Dense(1))
# 곱셈과 나눗셈용


model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.00029), metrics=['accuracy'])

# model.fit(train_data,train_ans,batch_size=1,epochs=20) #덧셈
model.fit(train_data, train_ans2, batch_size=1, epochs=20) #뺄셈
#model.fit(train_data, train_ans3, batch_size=1, epochs=20) #곱셈
#model.fit(train_data, train_ans4, batch_size=1, epochs=20) #나눗셈

z = np.array([4, 2]).reshape(1, 2)
q = model.predict(z, batch_size=1)
q= np.exp(q) #곱셈과 나눗셈용
print(q)

z = np.array([10, 20]).reshape(1, 2)
q = model.predict(z, batch_size=1)
q= np.exp(q)
print(q)

z = np.array([30, 20]).reshape(1, 2)
q = model.predict(z, batch_size=1)
q= np.exp(q)
print(q)

z = np.array([100, 120]).reshape(1, 2)
q = model.predict(z, batch_size=1)
q= np.exp(q)
print(q)