import numpy as np

import matplotlib.pyplot as plt

data_input = np.load('./data_input.npy')
data_target = np.load('./data_target.npy')

input_a = np.array(data_input)
target_a = np.array(data_target)

train_input = input_a
train_target = target_a
test_input = input_a
test_target = target_a

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    input_a, target_a, stratify=target_a, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

pred = kn.predict([[25, 200]])
distances, indexes = kn.kneighbors([[25, 200]])

plt.scatter(train_input[:, 0], train_input[:, 1], c=train_target)

plt.xlabel('length')
plt.ylabel('weight')
plt.show()

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std

new = ([25, 200] - mean) / std
plt.scatter(train_scaled[:, 0], train_scaled[:, 1], c=train_target)
plt.scatter(new[0], new[1], marker='^', c=pred)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)
print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1], c=train_target)
plt.scatter(new[0], new[1], marker='^', c=pred)
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


