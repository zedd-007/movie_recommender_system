import numpy as np
import pandas as pd
import ast
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv("C:\Users\Zaid Chikte\Desktop\mini project sem 5\Machine Learning Projects\Movie Recommendation System\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\Users\Zaid Chikte\Desktop\mini project sem 5\Machine Learning Projects\Movie Recommendation System\tmdb_5000_credits.csv")

movies.head(1)
credits.head(1)
movies = movies.merge(credits,on='title')

movies.head(1)
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head(1)
movies.isnull().sum()
movies.dropna(inplace=True)
movies.iloc[0].genres

def convert(obj):
    L = []
    import ast
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

movies.head()

movies['keywords'] = movies['keywords'].apply(convert)

movies.head()

movies['cast'][0]

def convert3(obj):
    L = []
    counter = 0
    import ast
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

movies.head()

movies['crew'][0]

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
        
    return L

movies['crew'].apply(fetch_director)

movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()

movies['overview'][0]

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df.head()

new_df['tags'][0]

new_df['tags'][1]

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors

vectors[0]

cv.get_feature_names_out()

ps = PorterStemmer()

def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:9]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

pickle.dump(new_df,open('movies.pkl','wb'))

new_df.to_dict()

pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))