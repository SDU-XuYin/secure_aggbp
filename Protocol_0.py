from Federated_learning import decrypt_vector, encrypt_vector
from sklearn.datasets import load_diabetes
from Federated_learning import decrypt_vector, sum_encrypted_vectors
import phe as paillier
import numpy as np

class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self, key_length):
         keypair = paillier.generate_paillier_keypair(n_length=key_length)
         self.pubkey, self.privkey = keypair

    def decrypt_aggregate(self, input_model, n_clients):
        return decrypt_vector(self.privkey, input_model) / n_clients


class Client:
    """Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.
    Using public key can encrypt locally computed gradients.
    """

    def __init__(self, name, X, y, pubkey):
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient

    def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

    def predict(self, X):
        """Score test data"""
        return X.dot(self.weights)#求内积

    def encrypted_gradient(self, sum_to=None):
        """Compute and encrypt gradient.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        gradient = self.compute_gradient()
        encrypted_gradient = encrypt_vector(self.pubkey, gradient)

        if sum_to is not None:
            return sum_encrypted_vectors(sum_to, encrypted_gradient)
        else:
            return encrypted_gradient
