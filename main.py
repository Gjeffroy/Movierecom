import numpy as np
import pandas as pd
from collections import defaultdict


# helper cosine similarity matrix
def csm(A, B):
    num = np.dot(A, B.T)
    p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
    p2 = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]
    return num/(p1*p2)


# add user to matrix 1 and 2 to have the same users in both matrices
def uniformize_userId(matrix1, matrix2):
    addto_1 = list(set(matrix2['userId']) - set(matrix1['userId']))
    # append at list one user otherwise matrix are not same size after pivoting
    addto_1.append(0)
    addto_1 = pd.DataFrame({'userId': addto_1,
                            'movieId': np.zeros(len(addto_1))})

    addto_2 = list(set(matrix1['userId']) - set(matrix2['userId']))
    # append at list one user otherwise matrix are not same size after pivoting
    addto_2.append(0)
    addto_2 = pd.DataFrame({'userId': addto_2,
                            'movieId': np.zeros(len(addto_2))})

    matrix1 = pd.concat([matrix1, addto_1]).sort_values(
        by=['userId', 'movieId']).reset_index(drop=True)
    matrix2 = pd.concat([matrix2, addto_2]).sort_values(
        by=['userId', 'movieId']).reset_index(drop=True)

    return matrix1, matrix2


# load data
ratings = pd.read_csv('./data/ml-25m/ratings.csv', nrows=10000)
ratings.drop('timestamp', axis=1, inplace=True)

# get unique movie Id
movieid = ratings['movieId'].unique()
nb_movies = len(movieid)

# get unique user Id
userId = ratings['userId'].unique()
nb_users = len(userId)

# spliting data into chunk of movies
nb_chunk = 10
moviechunks = dict()
for i in range(nb_chunk):
    moviechunks[i] = movieid[(i)*int(nb_movies/nb_chunk)
                              : (i+1)*int(nb_movies/nb_chunk)]


similarity_big = pd.DataFrame()
for i, chunk1 in moviechunks.items():
    row_chunk = pd.DataFrame(index=chunk1)
    for j, chunk2 in moviechunks.items():

        # filter to get movies in chunk 1 and 2
        matrix1 = ratings[ratings['movieId'].isin(chunk1)]
        matrix2 = ratings[ratings['movieId'].isin(chunk2)]

        # max sure all user are the same in chunk 1 and 2
        # otherwise matrix cosine won't work
        matrix1, matrix2 = uniformize_userId(matrix1, matrix2)

        # pivot matrix 1 and 2
        matrix1 = matrix1.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        matrix2 = matrix2.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        
        #drop dummy movies 0
        matrix1 = matrix1.drop(0)
        matrix2 = matrix2.drop(0)
        matrix1 = matrix1[matrix1.index!=0]
        matrix2 = matrix2[matrix2.index!=0]

        # compute cosine similarity on the chunk
        similarity = csm(matrix1, matrix2)
        similarity = pd.DataFrame(
            similarity, index=matrix1.index, columns=matrix2.index)
        print(similarity)

        row_chunk = row_chunk.join(similarity)

    if len(similarity_big)==0:
        similarity_big = row_chunk
    else:
        similarity_big = pd.concat([similarity_big, row_chunk])
    
    


    

#  top10[i][j] = {"similarity" :similarity.apply(
#             lambda x: np.argpartition(x, -10), axis=1).to_numpy(),
#             "movieId":matrix1.index.to_numpy().astype(int)
#         }