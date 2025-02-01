# Community-Notes

Matrix Factorization:
The current implementation utilizes only one latent dimension for the viewpoint agreement term, meant to represent political polarity. What happens if we add a second dimension? Matrix Factorization gets better at predicting with there are more latent dimensions. We don’t know what these dimensions correspond to.

Unsupervised learning is useful in this domain to identify patterns in human usage behavior. 

We can mitigate reliance on too many features (or find better features), by incrementally removing features as long as the distinct clustering is preserved.

Less “things to worry about” = less complexity when modeling known preferences and predicting future preferences.
