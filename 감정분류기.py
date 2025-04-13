from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 학습 데이터 준비
texts = ["너무 좋아", "기분이 최고야", "정말 짜증나", "별로야", "완전 행복해", "화가 나"]
labels = ["긍정", "긍정", "부정", "부정", "긍정", "부정"]

# 텍스트를 숫자로 변환
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 분류기 학습
model = MultinomialNB()
model.fit(X, labels)

# 예측
test = ["아니 오늘 래더하는데 자꾸 뭐라 하는거야 그래서 화가 나"]
test_X = vectorizer.transform(test)
print(model.predict(test_X))
