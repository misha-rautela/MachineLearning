#Use Case: Binary classification.
#Example: Spam detection (whether an email is spam or not).

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Binary classification (class '0' vs. others)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Output accuracy
print("Accuracy:", model.score(X_test, y_test))
