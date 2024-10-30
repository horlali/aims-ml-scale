import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Load and process ratings.csv
def load_ratings(
    filename,
) -> Tuple[
    Dict[int, List[Tuple[int, float]]],
    Dict[int, List[Tuple[int, float]]],
]:
    """
    Load and process ratings from a CSV file.

    Args:
        filename (str): The path to the CSV file containing the ratings.

    Returns:
        Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
            A tuple containing two dictionaries:
            - user_ratings: A dictionary where the key is the user ID and the value is a list of tuples,
              each containing a movie ID and the corresponding rating.
            - movie_ratings: A dictionary where the key is the movie ID and the value is a list of tuples,
              each containing a user ID and the corresponding rating.
    """
    user_ratings = defaultdict(list)  # Dictionary to store user-based ratings
    movie_ratings = defaultdict(list)  # Dictionary to store movie-based ratings

    print(user_ratings)

    with open(filename, "r") as file:
        next(file)  # Skip header
        row_count = 0
        for line in file:
            if row_count >= 1:
                break
            user_id, movie_id, rating, timestamp = line.strip().split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)

            # Append rating data to both user and movie-based dictionaries
            user_ratings[user_id].append((movie_id, rating))
            movie_ratings[movie_id].append((user_id, rating))

    return user_ratings, movie_ratings


# Load and process movies.csv
def load_movies(filename) -> Dict[int, Dict[str, List[str]]]:
    """
    Load and process movie information from a CSV file.

    Args:
        filename (str): The path to the CSV file containing the movie information.

    Returns:
        Dict[int, Dict[str, List[str]]]:
            A dictionary where the key is the movie ID and the value is another dictionary with:
            - "title": The title of the movie.
            - "genres": A list of genres associated with the movie.
    """
    movie_info = {}

    with open(filename, "r") as file:
        next(file)  # Skip header
        for line in file:
            movie_id, title, genres = line.strip().split(",", 2)
            movie_id = int(movie_id)
            movie_info[movie_id] = {"title": title, "genres": genres.split("|")}

    return movie_info


# Usage
user_ratings, movie_ratings = load_ratings("dataset_small/ratings.csv")
movie_info = load_movies("dataset_small/movies.csv")


# Extract all ratings from user_ratings
def get_all_ratings(user_ratings: Dict[int, List[Tuple[int, float]]]) -> np.ndarray:
    """
    Extract all ratings from the user_ratings dictionary.

    Args:
        user_ratings (Dict[int, List[Tuple[int, float]]]): A dictionary where the key is the user ID and the value is a list of tuples,
                                                           each containing a movie ID and the corresponding rating.

    Returns:
        np.ndarray: An array of all ratings.
    """
    all_ratings = []

    for ratings in user_ratings.values():
        for _, rating in ratings:
            all_ratings.append(rating)

    return np.array(all_ratings)


# Plot the rating distribution
def plot_rating_distribution(ratings):
    plt.figure(figsize=(8, 5))
    plt.hist(ratings, bins=np.arange(0.5, 5.5, 1), edgecolor="black", alpha=0.7)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(1, 6, 1))
    plt.show()


# Extract ratings and plot
all_ratings = get_all_ratings(user_ratings)
# plot_rating_distribution(all_ratings)


# Count the number of ratings per movie
def count_movie_ratings(movie_ratings):
    rating_counts = [len(ratings) for ratings in movie_ratings.values()]
    return np.array(rating_counts)


# Plotting the distribution on a log-log scale
def plot_power_law_distribution(rating_counts):
    plt.figure(figsize=(8, 5))
    plt.loglog(
        np.sort(rating_counts)[::-1],
        marker="o",
        linestyle="none",
        markersize=5,
        alpha=0.7,
    )
    plt.title("Log-Log Plot of Movie Rating Counts")
    plt.xlabel("Movie Rank (sorted by number of ratings)")
    plt.ylabel("Number of Ratings")
    plt.grid(True)
    plt.show()


# Count ratings per movie and plot
movie_rating_counts = count_movie_ratings(movie_ratings)
# plot_power_law_distribution(movie_rating_counts)


# Function to split data into train and test sets
def train_test_split(user_ratings, test_ratio=0.2):
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    for user, ratings in user_ratings.items():
        n_test = int(len(ratings) * test_ratio)
        test_ratings = random.sample(ratings, n_test)
        train_ratings = [rating for rating in ratings if rating not in test_ratings]

        train_data[user].extend(train_ratings)
        test_data[user].extend(test_ratings)

    return train_data, test_data


# RMSE Calculation
def calculate_rmse(
    data: Dict[int, List[Tuple[int, float]]],
    model_predict: Callable[[int, int], float],
) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) for the given data and prediction model.

    Args:
        data (Dict[int, List[Tuple[int, float]]]): A dictionary where the key is the user ID and the value is a list of tuples,
                                                   each containing a movie ID and the actual rating.
        model_predict (Callable[[int, int], float]): A function that takes a user ID and a movie ID and returns the predicted rating.

    Returns:
        float: The calculated RMSE value.
    """

    squared_errors = []

    for user, ratings in data.items():
        for movie_id, actual_rating in ratings:
            predicted_rating = model_predict(user, movie_id)
            squared_errors.append((predicted_rating - actual_rating) ** 2)

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse


# Dummy prediction function (for testing purposes)
def dummy_predict(user, movie_id):
    return 3.0  # Placeholder: Predicting an average rating of 3.0


# Split data into train and test sets
train_data, test_data = train_test_split(user_ratings)

# Calculate RMSE on train and test sets
train_rmse = calculate_rmse(train_data, dummy_predict)
test_rmse = calculate_rmse(test_data, dummy_predict)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)


def als(
    user_ratings: Dict[int, List[Tuple[int, float]]],
    n_users: int,
    n_movies: int,
    n_factors: int = 10,
    n_iterations: int = 10,
    lambda_reg: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Alternating Least Squares (ALS) to factorize the user-item rating matrix.

    Args:
        user_ratings (Dict[int, List[Tuple[int, float]]]): A dictionary where the key is the user ID and the value is a list of tuples,
                                                           each containing a movie ID and the corresponding rating.
        n_users (int): The number of users.
        n_movies (int): The number of movies.
        n_factors (int, optional): The number of latent factors. Defaults to 10.
        n_iterations (int, optional): The number of iterations to run the ALS algorithm. Defaults to 10.
        lambda_reg (float, optional): The regularization parameter. Defaults to 0.1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the user feature matrix (U) and the movie feature matrix (V).
    """
    # Initialize user and movie feature matrices with random values
    np.random.seed(0)
    U = np.random.normal(scale=1.0 / n_factors, size=(n_users, n_factors))
    V = np.random.normal(scale=1.0 / n_factors, size=(n_movies, n_factors))

    # Build user-movie rating matrix (sparse)
    R = np.zeros((n_users, n_movies))
    for user, ratings in user_ratings.items():
        for movie_id, rating in ratings:
            R[user - 1, movie_id - 1] = rating  # Adjust for zero-based indexing

    for iteration in range(n_iterations):
        # Fix V, update U
        for u in range(n_users):
            # Get movie indices and ratings for this user
            movies_rated_by_user = R[u, :] > 0
            V_u = V[movies_rated_by_user]
            ratings_u = R[u, movies_rated_by_user]

            # Solve for U[u] using least squares
            A = V_u.T.dot(V_u) + lambda_reg * np.eye(n_factors)
            B = V_u.T.dot(ratings_u)
            U[u] = np.linalg.solve(A, B)

        # Fix U, update V
        for i in range(n_movies):
            # Get user indices and ratings for this movie
            users_rated_movie = R[:, i] > 0
            U_i = U[users_rated_movie]
            ratings_i = R[users_rated_movie, i]

            # Solve for V[i] using least squares
            A = U_i.T.dot(U_i) + lambda_reg * np.eye(n_factors)
            B = U_i.T.dot(ratings_i)
            V[i] = np.linalg.solve(A, B)

        # Compute training RMSE for current iteration
        train_rmse = calculate_rmse(
            user_ratings, lambda u, i: np.dot(U[u - 1], V[i - 1])
        )
        print(f"Iteration {iteration + 1}: Training RMSE = {train_rmse:.4f}")

    return U, V


# Define number of users and movies based on dataset
n_users = max(user_ratings.keys())
n_movies = max(movie_info.keys())

# Run ALS and train model
U, V = als(
    user_ratings, n_users, n_movies, n_factors=10, n_iterations=10, lambda_reg=0.1
)


# Define prediction function using learned U and V
def model_predict(user, movie):
    return np.dot(U[user - 1], V[movie - 1])


# Calculate RMSE on train and test sets
train_rmse = calculate_rmse(train_data, model_predict)
test_rmse = calculate_rmse(test_data, model_predict)

print("Final Training RMSE:", train_rmse)
print("Final Test RMSE:", test_rmse)


# Define hyperparameter ranges
latent_factors = [5, 10, 20]  # Example values for K
regularization_params = [0.01, 0.1, 1.0]  # Example values for lambda

# Store results for each hyperparameter combination
results = []

# Perform grid search
for k in latent_factors:
    for lambda_reg in regularization_params:
        print(f"Training model with K={k}, lambda={lambda_reg}")

        # Train ALS model with current hyperparameters
        U, V = als(
            train_data,
            n_users,
            n_movies,
            n_factors=k,
            n_iterations=10,
            lambda_reg=lambda_reg,
        )

        # Calculate RMSE on training and test sets
        train_rmse = calculate_rmse(
            train_data, lambda user, movie: np.dot(U[user - 1], V[movie - 1])
        )
        test_rmse = calculate_rmse(
            test_data, lambda user, movie: np.dot(U[user - 1], V[movie - 1])
        )

        # Store results
        results.append((k, lambda_reg, train_rmse, test_rmse))
        print(
            f"K={k}, lambda={lambda_reg} => Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}\n"
        )

# Display results sorted by test RMSE
sorted_results = sorted(results, key=lambda x: x[3])  # Sort by test RMSE (index 3)
for k, lambda_reg, train_rmse, test_rmse in sorted_results:
    print(
        f"K={k}, lambda={lambda_reg} | Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}"
    )
