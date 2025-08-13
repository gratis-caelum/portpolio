# Python을 활용한 데이터 분석 기초

Python은 데이터 분석 분야에서 가장 널리 사용되는 프로그래밍 언어 중 하나입니다. 이번 포스트에서는 Python의 주요 데이터 분석 라이브러리인 `pandas`와 `numpy`를 활용한 기본적인 데이터 분석 방법을 알아보겠습니다.

## 필요한 라이브러리 설치

먼저 필요한 라이브러리들을 설치해보겠습니다.

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## 데이터 분석의 기본 단계

데이터 분석은 일반적으로 다음과 같은 단계를 거칩니다:

1. **데이터 수집**: 분석할 데이터 확보
2. **데이터 탐색**: 데이터의 구조와 특성 파악
3. **데이터 정제**: 결측값, 이상값 처리
4. **데이터 분석**: 통계적 분석 및 패턴 발견
5. **결과 시각화**: 분석 결과를 그래프로 표현

## Pandas 기본 사용법

### 데이터프레임 생성

```python
import pandas as pd
import numpy as np

# 딕셔너리로부터 데이터프레임 생성
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['Seoul', 'Busan', 'Seoul', 'Incheon']
}

df = pd.DataFrame(data)
print(df)
```

### 데이터 탐색

```python
# 기본 정보 확인
print(df.info())
print(df.describe())
print(df.head())

# 컬럼별 유니크 값 확인
print(df['city'].value_counts())
```

### 데이터 필터링

```python
# 조건에 따른 필터링
seoul_residents = df[df['city'] == 'Seoul']
adults = df[df['age'] >= 30]

# 복합 조건
young_seoul = df[(df['age'] < 30) & (df['city'] == 'Seoul')]
```

## Numpy를 활용한 수치 계산

```python
import numpy as np

# 배열 생성
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# 기본 통계 함수
print(f"평균: {np.mean(arr)}")
print(f"표준편차: {np.std(arr)}")
print(f"최댓값: {np.max(arr)}")
print(f"최솟값: {np.min(arr)}")

# 배열 연산
arr_squared = arr ** 2
normalized = (arr - np.mean(arr)) / np.std(arr)
```

## 데이터 시각화

시각화를 위해 `matplotlib`과 `seaborn`을 사용할 수 있습니다.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# 기본 그래프
ages = df['age']
plt.figure(figsize=(8, 6))
plt.hist(ages, bins=5, alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 도시별 인구 분포
city_counts = df['city'].value_counts()
plt.figure(figsize=(8, 6))
city_counts.plot(kind='bar')
plt.title('Population by City')
plt.xticks(rotation=45)
plt.show()
```

## 실제 데이터셋으로 연습하기

실제 데이터셋을 활용한 간단한 분석 예제입니다:

```python
# CSV 파일 읽기
# df = pd.read_csv('your_dataset.csv')

# 예시 판매 데이터 생성
sales_data = {
    'date': pd.date_range('2024-01-01', periods=100),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'sales': np.random.randint(10, 100, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}

sales_df = pd.DataFrame(sales_data)

# 월별 매출 집계
sales_df['month'] = sales_df['date'].dt.month
monthly_sales = sales_df.groupby('month')['sales'].sum()

print("월별 총 매출:")
print(monthly_sales)

# 제품별 평균 매출
product_avg = sales_df.groupby('product')['sales'].mean()
print("\n제품별 평균 매출:")
print(product_avg)
```

## 마무리

이번 포스트에서는 Python을 활용한 데이터 분석의 기초를 다뤄봤습니다. `pandas`와 `numpy`는 데이터 분석의 핵심 도구이므로, 다양한 데이터셋으로 연습해보시기 바랍니다.

다음 포스트에서는 더 고급 데이터 분석 기법과 머신러닝 기초에 대해 다뤄보겠습니다.

### 참고 자료

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

궁금한 점이 있으시면 언제든 연락 주세요! 📊