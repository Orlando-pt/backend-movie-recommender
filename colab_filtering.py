import numpy as np
import pandas as pd
import collections
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

# Add some convenience functions to Pandas DataFrame.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format

def mask(df, key, function):
  """Returns a filtered dataframe, by applying function to key"""
  return df[function(df[key])]

def flatten_cols(df):
  df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df

pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

from cf_model import CFModel

## util functions
def mark_genres(movies, genres):
    def get_random_genre(gs):
        active = [genre for genre, g in zip(genres, gs) if g==1]
        if len(active) == 0:
            return 'Other'
        return np.random.choice(active)
    def get_all_genres(gs):
        active = [genre for genre, g in zip(genres, gs) if g==1]
        if len(active) == 0:
            return 'Other'
        return '-'.join(active)
    movies['genre'] = [
        get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
    movies['all_genres'] = [
        get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]

def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
        df: a dataframe.
        holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
        train: dataframe for training
        test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test

def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return 1. / (U.shape[0]*V.shape[0]) * tf.reduce_sum(
        tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))


# global variables
DOT = 'dot'
COSINE = 'cosine'

class CF:
    def __init__(self):
        self.users = None
        self.movies = None
        self.ratings = None

        self.model = None       # has the current trained model

        self.load_data()
    
    @property
    def get_movies(self):
        return self.movies.to_json(orient='records')

    def load_data(self):
        # Load each data set (users, movies, and ratings).
        users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.users = pd.read_csv(
            'ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

        ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings = pd.read_csv(
            'ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

        # The movies file contains a binary feature for each genre.
        genre_cols = [
            "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        movies_cols = [
            'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
        ] + genre_cols
        self.movies = pd.read_csv(
            'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

        # Since the ids start at 1, we shift them to start at 0.
        self.users["user_id"] = self.users["user_id"].apply(lambda x: str(x-1))
        self.movies["movie_id"] = self.movies["movie_id"].apply(lambda x: str(x-1))
        self.movies["year"] = self.movies['release_date'].apply(lambda x: str(x).split('-')[-1])
        self.ratings["movie_id"] = self.ratings["movie_id"].apply(lambda x: str(x-1))
        self.ratings["user_id"] = self.ratings["user_id"].apply(lambda x: str(x-1))
        self.ratings["rating"] = self.ratings["rating"].apply(lambda x: float(x))

        mark_genres(self.movies, genre_cols)

    def add_user_ratings(self, movies, ratings):
        my_ratings = pd.DataFrame({
            'user_id': "943",
            'movie_id': list(map(str, movies)),
            'rating': list(map(float, ratings)),
        })
        # Remove previous ratings.
        self.ratings = self.ratings[self.ratings.user_id != "943"]
        # Add new ratings.
        self.ratings = self.ratings.append(my_ratings, ignore_index=True)
        # Add new user to the users DataFrame.
        if self.users.shape[0] == 943:
            self.users = self.users.append(self.users.iloc[942], ignore_index=True)
            self.users["user_id"][943] = "943"
        print("Added your %d ratings; you have great taste!" % len(my_ratings))
        self.ratings[self.ratings.user_id=="943"].merge(self.movies[['movie_id', 'title']])

    def compute_scores(self, query_embedding, item_embeddings, measure=DOT):
        """Computes the scores of the candidates given a query.
        Args:
            query_embedding: a vector of shape [k], representing the query embedding.
            item_embeddings: a matrix of shape [N, k], such that row i is the embedding
            of item i.
            measure: a string specifying the similarity measure to be used. Can be
            either DOT or COSINE.
        Returns:
            scores: a vector of shape [N], such that scores[i] is the score of item i.
        """
        u = query_embedding
        V = item_embeddings
        if measure == COSINE:
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            u = u / np.linalg.norm(u)
        scores = u.dot(V.T)
        return scores

    def user_recommendations(self, model, measure=DOT, exclude_rated=True, k=6):
        scores = self.compute_scores(
            model.embeddings["user_id"][943], model.embeddings["movie_id"], measure)
        score_key = measure + '_score'
        df = pd.DataFrame({
            score_key: list(scores),
            'movie_id': self.movies['movie_id'],
            'titles': self.movies['title'],
            'genres': self.movies['all_genres'],
        })
        if exclude_rated:
            # remove movies that are already rated
            rated_movies = self.ratings[self.ratings.user_id == "943"]["movie_id"].values
            df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]

        ret = df.sort_values([score_key], ascending=False).head(k)
        print(ret)
        return ret
    
    def movie_neighbors(self, model, title_substring, measure=DOT, k=6):
        # Search for movie ids that match the given substring.
        ids =  self.movies[self.movies['title'].str.contains(title_substring)].index.values
        titles = self.movies.iloc[ids]['title'].values
        if len(titles) == 0:
            raise ValueError("Found no movies with title %s" % title_substring)
        print("Nearest neighbors of : %s." % titles[0])
        if len(titles) > 1:
            print("[Found more than one matching movie. Other candidates: {}]".format(
                ", ".join(titles[1:])))
        movie_id = ids[0]
        scores = self.compute_scores(
            model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
            measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'titles': self.movies['title'],
            'genres': self.movies['all_genres']
        })

        ret = df.sort_values([score_key], ascending=False).head(k)
        print(ret)
        return ret

    def build_rating_sparse_tensor(self, ratings_df):
        """
        Args:
            ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
        Returns:
            a tf.SparseTensor representing the ratings matrix.
        """
        indices = ratings_df[['user_id', 'movie_id']].values
        values = ratings_df['rating'].values
        return tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[self.users.shape[0], self.movies.shape[0]])

    def sparse_mean_square_error(self, sparse_ratings, user_embeddings, movie_embeddings):
        """
        Args:
            sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
            user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
            dimension, such that U_i is the embedding of user i.
            movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
            dimension, such that V_j is the embedding of movie j.
        Returns:
            A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = tf.reduce_sum(
            tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
            tf.gather(movie_embeddings, sparse_ratings.indices[:, 1]),
            axis=1)
        loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
        return loss

    def build_regularized_model(
            self, ratings, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
            init_stddev=0.1):
        """
        Args:
            ratings: the DataFrame of movie ratings.
            embedding_dim: The dimension of the embedding space.
            regularization_coeff: The regularization coefficient lambda.
            gravity_coeff: The gravity regularization coefficient lambda_g.
        Returns:
            A CFModel object that uses a regularized loss.
        """
        # Split the ratings DataFrame into train and test.
        train_ratings, test_ratings = split_dataframe(ratings)
        # SparseTensor representation of the train and test datasets.
        A_train = self.build_rating_sparse_tensor(train_ratings)
        A_test = self.build_rating_sparse_tensor(test_ratings)
        U = tf.Variable(tf.random_normal(
            [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
        V = tf.Variable(tf.random_normal(
            [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

        error_train = self.sparse_mean_square_error(A_train, U, V)
        error_test = self.sparse_mean_square_error(A_test, U, V)
        gravity_loss = gravity_coeff * gravity(U, V)
        regularization_loss = regularization_coeff * (
            tf.reduce_sum(U*U)/U.shape[0] + tf.reduce_sum(V*V)/V.shape[0])
        total_loss = error_train + regularization_loss + gravity_loss
        losses = {
            'train_error_observed': error_train,
            'test_error_observed': error_test,
        }
        loss_components = {
            'observed_loss': error_train,
            'regularization_loss': regularization_loss,
            'gravity_loss': gravity_loss,
        }
        embeddings = {"user_id": U, "movie_id": V}

        return CFModel(embeddings, total_loss, [losses, loss_components])

    def build_model(self):
        self.model = self.build_regularized_model(
            self.ratings, regularization_coeff=0.1, gravity_coeff=1.0, embedding_dim=35,
            init_stddev=.05)
    
    def train_model(self):
        self.model.train(num_iterations=2000, learning_rate=20.)

    def get_movie_neighbors(self, movie, measure=DOT):
        if not self.model:
            RuntimeError('Model not yet initialized.')
            return None
        return self.movie_neighbors(self.model, movie, measure)

    def get_user_recommendations(self, measure, nr_recommendations=10):
        if not self.model:
            RuntimeError('Model not yet initialized.')
            return None
        return self.user_recommendations(self.model, measure, k=nr_recommendations).to_json(orient='records')


if __name__ == '__main__':
    app = CF()

    app.add_user_ratings(
        ["0", "10", "21", "26", "55", "63", "68"],
        [4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    )

    app.build_model()
    app.train_model()

    app.get_user_recommendations(DOT)
    app.get_user_recommendations(COSINE)

    # app.get_movie_neighbors("Aladdin")
    # app.get_movie_neighbors("Aladdin", COSINE)
