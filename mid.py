import pandas as pd

mid = pd.read_csv("midterm.csv")
#print(mid.shape)
#print(mid.head())
#print(pd.unique(mid['Species']))

mid_input = mid[['Weight','Length','Diagonal','Height','Width']].to_numpy()
#print(mid_input[:5])
mid_target = mid['Species'].to_numpy()
import numpy as np
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    mid_input, mid_target, random_state=42
)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

#테스트 세트 처음 5개 샘플 예측
#print(lr.predict(test_scaled[:5]))
#proba = lr.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=3))

#분류 특성 열 확인
#print(lr.classes_)
#print(lr.coef_.shape, lr.intercept_.shape)

#소프트 맥스 함수
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))

# Width를 제외한 4가지 특성을 조합하는 것이 가장 정확도가 높았다.