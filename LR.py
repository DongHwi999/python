import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

fp = pd.read_excel("target.xlsx", header=None)
ep = pd.read_excel("input.xlsx", header=None)

#입력데이터 확인하기
#ep.plot()
#plt.show()

#1열의 스케일 값이 너무 작아 가중치 분석하기 힘드므로 1열 삭제
ep = ep.iloc[:, 1:]

X = ep.iloc[:, :2]
y = fp.iloc[:, 0]

# 학습데이터와 테스트데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
lr = LinearRegression()
lr.fit(X_train, y_train)

#print(lr.coef_, lr.intercept_)
#plt.scatter(X_train, y_train)
#plt.show()

# 학습용 데이터 정확도 출력
train_score = lr.score(X_train, y_train)
print("First Train Score:", train_score)

# 테스트 데이터로 모델 성능 평가
y_poly = lr.predict(X_test)
score = r2_score(y_test, y_poly)
print("First Test Score:", score)

# 정확도를 올리기 위해 2차 방정식으로 전환
X_train = np.array(X_train)
X_train_poly = np.column_stack((X_train ** 2, X_train))

X_test = np.array(X_test)
X_test_poly = np.column_stack((X_test ** 2, X_test))

#print(X_train_poly.shape, X_test_poly.shape)

lr.fit(X_train_poly, y_train)

#파라미터 조정 후 학습용 데이터 정확도 출력
train_score = lr.score(X_train_poly, y_train)
print("Second Train Score:", train_score)

# 테스트 데이터로 모델 성능 평가
y_pred = lr.predict(X_test_poly)
test_score = r2_score(y_test, y_pred)
print("Second Test Score:", test_score)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train_poly)
train_scaled = ss.transform(X_train_poly)
test_scaled = ss.transform(X_test_poly)

# 모델 학습
rr = Ridge(alpha=1) # alpha 값을 조절하여 규제 강도 조정
rr.fit(train_scaled, y_train)

#print(rr.coef_, rr.intercept_)
#plt.scatter(X_train, y_train)
#plt.show()

# 학습용 데이터 정확도 출력
train_score = rr.score(train_scaled, y_train)
print("Train Score:", train_score)

# 테스트 데이터로 모델 성능 평가
y_pred = rr.predict(X_test)
test_score = r2_score(y_test, y_pred)
print("Test Score:", test_score)



