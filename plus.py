import tensorflow as tf
import numpy as np

# 학습 데이터 생성
num_samples = 100000
num_features = 1
num_hidden = 2
num_classes = 1

np.random.seed(0)
Num1 = np.random.rand(num_samples, num_features)
Num2 = np.random.rand(num_samples, num_features)
Ans = Num1 + Num2

# MLP 모델 구성
inputs = tf.keras.Input(shape=(num_features,))
hidden = tf.keras.layers.Dense(num_hidden, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(num_classes)(hidden)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse')

# 학습
model.fit(x=Num1, y=Ans, epochs=10, batch_size=64)

# 평가
test_Num1 = 10
test_Num2 = 20
test_Ans = test_Num1 + test_Num2
print("test_Num1 + test_Num2 = test_Ans ", test_Num1, test_Num2, test_Ans )

test_loss = model.evaluate(x=test_Num1, y=test_Ans)
print("Test loss:", test_loss)

