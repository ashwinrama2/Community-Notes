#@title MSE
import numpy as np
import pandas as pd
from math import nan


f = open("loss.txt", "w")
f.write("")
f.close()


# Parameters
num_users = 25  # Number of users (rows in the matrix)
num_notes = 15  # Number of notes (columns in the matrix)
latent_dim = 1  # Dimensionality of latent factors (k)
lambda_i = 0.15  # Regularization for intercepts
lambda_f = 0.03  # Regularization for latent factors
learning_rate = 0.00001  # Learning rate for gradient descent
iterations = 100000  # Total iterations
print_every = 100  # Print loss every n iterations

# Initialize the matrix with random ratings (0, 1, or None)
#np.random.seed(42)
ratings = np.random.choice([0, 0.5, 1, None], size=(num_users, num_notes), p=[0.2, 0.1, 0.2, 0.5])

# Convert the initial matrix to a readable format
initial_ratings = np.array([
    [1.0 if r == 1 else 0.5 if r == 0.5 else 0.0 if r == 0 else np.nan for r in row]
    for row in ratings
])

initial_ratings_df = pd.DataFrame(
    initial_ratings,
    columns=[f"Note {n+1}" for n in range(num_notes)],
    index=[f"User {u+1}" for u in range(num_users)]
)

print("Initial Ratings Matrix:")
print(initial_ratings_df)

# Initialize parameters
mu = np.random.rand()  # Global intercept
i_u = np.random.rand(num_users)  # User-specific intercepts
i_n = np.random.rand(num_notes)  # Note-specific intercepts
f_u = np.random.rand(num_users, latent_dim)  # User factors: num_users x latent_dim
f_n = np.random.rand(num_notes, latent_dim)  # Note factors: num_notes x latent_dim

# Print initial intercepts
print("\nInitial Intercepts:")
print(f"Global Intercept (mu): {mu:.4f}")
print(f"User Intercepts (i_u): {i_u}")
print(f"Note Intercepts (i_n): {i_n}")

# Training with gradient descent
losses = []
for iteration in range(iterations):
    total_loss = 0

    # Iterate over all ratings
    for u in range(num_users):
        for n in range(num_notes):
            if ratings[u, n] is not None:  # Only update for observed ratings
                r_un = ratings[u, n]
                r_hat_un = mu + i_u[u] + i_n[n] + np.dot(f_u[u], f_n[n])

                # Compute error
                error = r_un - r_hat_un

                # Accumulate loss
                total_loss += error ** 2

                # Gradients for each parameter
                grad_mu = -2 * error + 2 * lambda_i * mu
                grad_i_u = -2 * error + 2 * lambda_i * i_u[u]
                grad_i_n = -2 * error + 2 * lambda_i * i_n[n]
                grad_f_u = -2 * error * f_n[n] + 2 * lambda_f * f_u[u]
                grad_f_n = -2 * error * f_u[u] + 2 * lambda_f * f_n[n]

                # Update parameters
                mu -= learning_rate * grad_mu
                i_u[u] -= learning_rate * grad_i_u
                i_n[n] -= learning_rate * grad_i_n
                f_u[u] -= learning_rate * grad_f_u
                f_n[n] -= learning_rate * grad_f_n

    # Add regularization to the loss
    total_loss += lambda_i * (mu ** 2 + np.sum(i_u ** 2) + np.sum(i_n ** 2))
    total_loss += lambda_f * (np.sum(f_u ** 2) + np.sum(f_n ** 2))
    losses.append(total_loss)

    # Print loss at intervals
    if (iteration + 1) % print_every == 0 or iteration == 0:
        f = open("loss.txt", "a")
        f.write(str(total_loss) + "\n")
        f.close()
        print(f"Iteration {iteration + 1}: Loss = {total_loss:.4f}")

# Final predicted values
predicted_ratings = np.zeros_like(ratings, dtype=float)
for u in range(num_users):
    for n in range(num_notes):
        r_hat_un = mu + i_u[u] + i_n[n] + np.dot(f_u[u], f_n[n])
        predicted_ratings[u, n] = 1 if r_hat_un > 0.5 else 0

# Convert predicted matrix to readable format
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    columns=[f"Note {n+1}" for n in range(num_notes)],
    index=[f"User {u+1}" for u in range(num_users)]
)

# Print final intercepts
print("\nFinal Intercepts:")
print(f"Global Intercept (mu): {mu:.4f}")
print(f"User Intercepts (i_u): {i_u}")
print(f"Note Intercepts (i_n): {i_n}")

print("\nPredicted Ratings Matrix:")
print(predicted_ratings_df.round(2))

# Function to compute MSE loss between predicted and initial ratings
def diff(predicted_ratings, initial_ratings):
    valid_mask = ~np.isnan(initial_ratings)  # Mask for valid (non-NaN) entries
    valid_initial = initial_ratings[valid_mask]
    valid_predicted = predicted_ratings[valid_mask]
    mse_loss = np.mean((valid_initial - valid_predicted) ** 2)
    return mse_loss

# Compute and print the MSE loss
mse_loss = diff(predicted_ratings, initial_ratings)
print(f"\nMean Squared Error (MSE) between initial and predicted ratings: {mse_loss:.4f}")
