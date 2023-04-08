import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

fp = pd.read_excel("target.xlsx", header=None)
ep = pd.read_excel("input.xlsx", header=None)

#1열의 스케일 값이 너무 작아 가중치 분석하기 힘드므로 1열 삭제
ep = ep.iloc[:, 1:]

x = ep.iloc[:, :2]
y = fp.iloc[:, 0]

# 학습데이터와 테스트데이터 분리
train_input, test_input, train_target, test_target = train_test_split(x, y, test_size=0.2, random_state=42)

# 모델 학습
lr = LinearRegression()
lr.fit(train_input, train_target)

plt.scatter(train_input.iloc[:, 0], train_target)
plt.scatter(train_input.iloc[:, 1], train_target)
plt.xlabel('use')
plt.ylabel('a')
plt.show()
#print(lr.coef_, lr.intercept_)

# 학습용 데이터 정확도 출력
train_score1 = lr.score(train_input, train_target)
print("First Train Score:", train_score1)

# 테스트 데이터로 모델 성능 평가
test_score1 = lr.predict(test_input)
score = r2_score(test_target, test_score1)
print("First Test Score:", score)

# 정확도를 올리기 위해 다항회귀로 전환
train_input = np.array(train_input)
train_poly = np.column_stack((train_input ** 2, train_input))

test_input = np.array(test_input)
test_poly = np.column_stack((test_input ** 2, test_input))

#print(train_poly.shape, test_poly.shape)
#print(train_poly, test_poly)

lr.fit(train_poly, train_target)
#print(lr.coef_, lr.intercept_)

# 다항회귀 학습용 데이터 정확도 출력
train_score = lr.score(train_poly, train_target)
print("Second Train Score:", train_score)

# 테스트 데이터로 2번째 모델 성능 평가
y_pred = lr.predict(test_poly)
test_score = r2_score(test_target, y_pred)
print("Second Test Score:", test_score)