# -*- coding: utf-8 -*-
"""영화 추천 알고리즘.ipynb"""

# 데이터 전처리 과정
import pandas as pd

rating_data = pd.read_csv('/content/data/ratings.csv') # 영화-사용자 평점 기반 데이터
movie_data = pd.read_csv('/content/data/movies.csv') # 영화 정보 데이터

rating_data.head() # 데이터 구조 파악
movie_data.head()

rating_data.drop('timestamp', axis = 1, inplace = True) # 불필요한 데이터 제거
movie_data.drop('genres', axis = 1, inplace = True)

user_movie_data = pd.merge(rating_data, movie_data, on = 'movieId') # movieId라는 공통 컬럼을 통해 두 파일을 합침
user_movie_rating = user_movie_data.pivot_table('rating', index = 'userId', columns='title').fillna(0) # 유저가 평점을 매기지 않은 영화에는 0으로 채워줌

SVD = TruncatedSVD(n_components=12) # latent(잠재요인값) = 12
matrix = SVD.fit_transform(movie_user_rating) # SVD.fit_transform으로 변환하면 영화데이터가 12개의 어떤 요소 값을 가지게 됨
matrix.shape
matrix[0]

corr = np.corrcoef(matrix) # 상관관계 행렬 계산
corr2 = corr[:200, :200] # 상위 200개의 행과 열만 선택하여 새로운 상관관계 행렬 생성
plt.figure(figsize=(16, 10)) # 상관계수 히트맵 형태로 시각화
sns.heatmap(corr2)

movie_title = user_movie_rating.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Guardians of the Galaxy (2014)")
corr_coffey_hands  = corr[coffey_hands]
list(movie_title[(corr_coffey_hands >= 0.9)])[:50] # 상관계수가 높은 영화 추출


matrix = df_user_movie_ratings.values# matrix는 pivot_table을 pd.values속성을 사용하여 matrix로 만든 것
user_ratings_mean = np.mean(matrix, axis = 1) # user_ratings_mean은 사용자의 평균 평점
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1) # R_user_mean : 사용자-영화에 대해 사용자 평균 평점을 뺀 것
matrix

# 특이값 분해
# scipy에서 제공해주는 svd 사용 / scikit learn에서도 svd를 제공하지만, U, sigma, Vt를 반환하지 않아서 scipy에서 제공하는 svd 사용
# U 행렬, sigma 행렬, V 전치 행렬을 반환
U, sigma, Vt = svds(matrix_user_mean, k = 12)

# SVD로 분해한 행렬을 복원시키는 과정 필요
# U, Sigma, Vt의 내적을 수행하여 다시 원본 행렬로 복원 / 내적코드 = np.dot(np.dot(U, sigma), Vt)
# 거기에 사용자 평균을 더해줌
svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# 최종적으로 SVD 특이값 분해를 통해 행렬분해 기반으로 데이터 변경해줌
df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_movie_ratings.columns)
df_svd_preds.head()


# 알고리즘 구현 과정
import numpy as np

df_ratings  = pd.read_csv('/content/data/ratings.csv')
df_movies  = pd.read_csv('/content/data/movies.csv')

df_user_movie_ratings = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

df_user_movie_ratings.head()

matrix = df_user_movie_ratings.valuesmatrix = df_user_movie_ratings.values
user_ratings_mean = np.mean(matrix, axis = 1)
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)
matrix

matrix.shape
user_ratings_mean.shape
matrix_user_mean.shape

pd.DataFrame(matrix_user_mean, columns = df_user_movie_ratings.columns).head()

U, sigma, Vt = svds(matrix_user_mean, k = 12)

print(U.shape)
print(sigma.shape)
print(Vt.shape)

sigma.shape
sigma[0]
sigma[1]

svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_movie_ratings.columns)
df_svd_preds.head()
df_svd_preds.shape

def recommend_movies(df_svd_preds, user_id, ori_movies_df, ori_ratings_df, num_recommendations=5):
  user_row_number = user_id - 1
  sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
  user_data = ori_ratings_df[ori_ratings_df.userId == user_id]
  user_history = user_data.merge(ori_movies_df, on = 'movieId').sort_values(['rating'], ascending=False)

  recommendations = ori_movies_df[~ori_movies_df['movieId'].isin(user_history['movieId'])]
  recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
  recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]

  return user_history, recommendations

already_rated, predictions = recommend_movies(df_svd_preds, 330, df_movies, df_ratings, 10)
already_rated.head(10)
predictions
