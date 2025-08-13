# 머신러닝 입문: 첫 번째 모델 만들기

머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 인공지능의 한 분야입니다. 이번 포스트에서는 Python의 `scikit-learn` 라이브러리를 사용하여 첫 번째 머신러닝 모델을 만들어보겠습니다.

## 머신러닝 기본 개념

### 지도학습 vs 비지도학습

- **지도학습 (Supervised Learning)**: 입력과 정답이 있는 데이터로 학습
  - 분류 (Classification): 카테고리 예측
  - 회귀 (Regression): 연속적인 값 예측

- **비지도학습 (Unsupervised Learning)**: 정답 없는 데이터에서 패턴 발견
  - 군집화 (Clustering)
  - 차원 축소 (Dimensionality Reduction)

## 환경 설정

필요한 라이브러리를 설치해봅시다:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## 첫 번째 분류 모델: 붓꽃 분류

가장 유명한 머신러닝 예제인 붓꽃(Iris) 데이터셋을 사용해보겠습니다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 로드
iris = datasets.load_iris()
X = iris.data  # 특성 (꽃받침 길이, 너비, 꽃잎 길이, 너비)
y = iris.target  # 타겟 (붓꽃 종류)

print("특성 이름:", iris.feature_names)
print("클래스 이름:", iris.target_names)
print("데이터 shape:", X.shape)
```

### 데이터 탐색

```python
# 데이터프레임 생성
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())
print(df.describe())

# 클래스 분포 확인
print("클래스 분포:")
print(df['species'].value_counts())
```

### 데이터 시각화

```python
# 특성 간 관계 시각화
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='species', diag_kind='hist')
plt.show()

# 상관관계 히트맵
plt.figure(figsize=(8, 6))
correlation_matrix = df.iloc[:, :-2].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

## 모델 학습 및 평가

### 데이터 분할

```python
# 학습용과 테스트용 데이터 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
```

### 로지스틱 회귀 모델

```python
# 로지스틱 회귀 모델 생성 및 학습
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 예측
y_pred_log = log_reg.predict(X_test)

# 정확도 계산
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"로지스틱 회귀 정확도: {accuracy_log:.3f}")
```

### 랜덤 포레스트 모델

```python
# 랜덤 포레스트 모델 생성 및 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 예측
y_pred_rf = rf_model.predict(X_test)

# 정확도 계산
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"랜덤 포레스트 정확도: {accuracy_rf:.3f}")
```

## 모델 평가

### 혼동 행렬 (Confusion Matrix)

```python
# 랜덤 포레스트 모델의 혼동 행렬
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### 분류 리포트

```python
# 상세한 분류 리포트
print("분류 리포트:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))
```

### 특성 중요도

```python
# 특성 중요도 시각화
feature_importance = rf_model.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), 
           [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()
```

## 새로운 데이터 예측

```python
# 새로운 데이터 포인트로 예측해보기
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # 새로운 붓꽃 측정값

prediction = rf_model.predict(new_sample)
prediction_proba = rf_model.predict_proba(new_sample)

print(f"예측 결과: {iris.target_names[prediction[0]]}")
print("예측 확률:")
for i, prob in enumerate(prediction_proba[0]):
    print(f"  {iris.target_names[i]}: {prob:.3f}")
```

## 모델 성능 향상 팁

### 1. 교차 검증 (Cross Validation)

```python
from sklearn.model_selection import cross_val_score

# 5-fold 교차 검증
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"교차 검증 평균 정확도: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### 2. 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# 그리드 서치로 최적 파라미터 찾기
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 교차 검증 점수: {grid_search.best_score_:.3f}")
```

## 마무리

이번 포스트에서는 scikit-learn을 사용하여 첫 번째 머신러닝 모델을 만들어봤습니다. 주요 내용을 정리하면:

1. **데이터 탐색**: 데이터의 구조와 분포 파악
2. **모델 학습**: 여러 알고리즘으로 모델 훈련
3. **모델 평가**: 정확도, 혼동 행렬, 분류 리포트 활용
4. **예측**: 새로운 데이터에 대한 예측 수행

### 다음 단계

- 더 복잡한 데이터셋으로 연습하기
- 다양한 머신러닝 알고리즘 시도하기
- 특성 엔지니어링 기법 학습하기
- 모델 해석 기법 익히기

궁금한 점이 있으시면 언제든 연락 주세요! 🤖

### 참고 자료

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Kaggle Learn](https://www.kaggle.com/learn)