import numpy as np  
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Creating the Support Vector Machine from Scratch
class SVM:
    def __init__(self, learning_rate=0.01, epochs=100, C=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C  # Regularzisation Parameter
        self.w = None   # Weight
        self.b = None   # Bias

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(self.epochs):
            for i, x in enumerate(X):
                condition = y[i] * (np.dot(x, self.w) + self.b) >= 1

                if not condition:
                    self.w = self.w - self.learning_rate * (self.w - self.C * y[i] * x)
                    self.b = self.b - self.learning_rate * (self.C * y[i])

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def decision_function(self, X):
        return np.dot(X, self.w) + self.b
    
# Random Synthetic Data to Test The SVM

X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

y = np.where(y == 0, -1, 1)

svm = SVM(learning_rate=0.01, epochs=1000, C=1.0)
svm.fit(X,y)

plt.scatter(X[:, 0], X[:, -1], c = y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='autumn', alpha=0.3)

plt.contour(xx, yy, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])

# Plotting HyperPlanes
decision_values = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
decision_values = decision_values.reshape(xx.shape)
plt.contour(xx, yy, decision_values, colors='k', levels=[-1,0,1], alpha=0.3, linestyles=['--', '-', '--'])

# Plotting Support Vectors
support_vector_indicies = np.where(np.abs(svm.decision_function(X)) <= 1)[0]
plt.scatter(X[support_vector_indicies, 0], X[support_vector_indicies, 1], s=150, edgecolors='k', facecolors='none')

# Labelling Data Points
for i, (x, y_label) in enumerate(zip(X, y)):
    plt.text(x[0], x[1], str(y_label), color='blue' if y_label == 1 else 'red', fontsize=8,
             ha='center', va='center')
plt.show()