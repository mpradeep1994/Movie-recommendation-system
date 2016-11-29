
#   Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


# In[35]:

def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())



# In[36]:

def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]"""
    tokenize = [tokenize_string(movies['genres'][i]) for i in range(len(movies)) ]
    movies.insert(len(movies.columns), "tokens",tokenize, allow_duplicates=False)
    return movies
    pass


# In[37]:

def tf(i,d,term_frequency):
    return term_frequency[d][i]
def df(i,doc_frequency):
    return doc_frequency[i]
def max_ktf(d,term_frequency):
    return term_frequency[d].most_common(1)[0][1]
def tfidf(i,d,term_frequency,doc_frequency,N):
    return (1.0*tf(i,d,term_frequency)) /max_ktf(d,term_frequency) * math.log10(N/df(i,doc_frequency))


# In[38]:

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which. has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    
    N = len(movies)
    doc_frequency= defaultdict(int)
    term_frequency = {}
    for i in range(len(movies)):
        count = Counter(movies['tokens'][i])
        term_frequency[i] = count
        for j in count:
            doc_frequency[j] += 1
    vocab = {}
    idx = 0
    for i in sorted(doc_frequency):
        vocab[i] = idx
        idx += 1
    
    csr_list =[]
    for doc in sorted(term_frequency):
        data = []
        column = []
        for term in term_frequency[doc]:
            data.append(tfidf(term,doc,term_frequency,doc_frequency,N))
            column.append(vocab[term])
        row = [0]*len(data)
        csr_list.append(csr_matrix((data, (row, column)), shape=(1, len(vocab)),dtype='float64'))
    
    movies.insert(len(movies.columns), "features",csr_list, allow_duplicates=False)

    return movies,vocab            
            
        


# In[39]:


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


# In[40]:

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    """dema = np.sqrt(np.sum([x*x for x in  a.toarray()]))
    demb = np.sqrt(np.sum([x*x for x in  b.toarray()]))

    return ((a.toarray().dot(b.T.toarray()))/dema*demb)[0][0]"""
    
    upper = np.dot(a.toarray(),b.T.toarray())
    a_nor=np.linalg.norm(a.toarray())
    b_nor=np.linalg.norm(b.toarray())
    cos=upper/(a_nor*b_nor)
    return cos[0][0]
    #return ((a.toarray().dot(b.T.toarray()))/dema*demb)[0][0]
    
                                                        
            


# In[41]:

def weighted_average(feature_csr,user_rated_ids,ratings,movies):
    cosines = []
    rate = ratings.tolist()
    rate1=[]
    
    for j in range(len(user_rated_ids)):
        b = (movies[movies['movieId']==user_rated_ids[j]]['features'].values)[0]
        cosi = cosine_sim(feature_csr,b)
        if cosi >= 0:    
            cosines.append(cosi)
            rate1.append(rate[j])
    
    if np.sum(cosines) > 0:
        return np.sum(np.array(cosines)*np.array(rate1))/np.sum(cosines)
    else:
        return np.mean(ratings)


# In[42]:

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    
    user_id  =ratings_test['userId'].values.tolist()
    movie_id  = ratings_test['movieId'].values.tolist()
    
    predict  = []
    for (u,v) in zip(user_id,movie_id):
        movies_rated_id = ratings_train[ratings_train['userId']==u]['movieId'].values.tolist() 
        csr_predict = (movies[movies['movieId']==v]['features'].values)[0]
        ratings = ratings_train[ratings_train['userId']==u]['rating']
        predict.append(weighted_average(csr_predict,movies_rated_id,ratings,movies))
    return np.array(predict)
    pass


# In[43]:


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


# In[44]:

def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    #print (movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])

if __name__ == '__main__':
    main()


# In[ ]:




# In[ ]:



