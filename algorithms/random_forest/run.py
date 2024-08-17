from .random_forest import RandomForest
from ..eval import train_test_split, accuracy, balanced_accuracy

## dataset -> load your dataset here
## get X and y -> X = dataset[:, :-1], y = dataset [:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)


#create model instance
model = RandomForest(n_trees=10, max_depth=10, min_samples=2)

# Fit the decision tree model to the training data.
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data.
predictions = model.predict(X_test)

# Calculate evaluating metrics
print(f"Model's Accuracy: {accuracy(y_test, predictions)}")
print(f"Model's Balanced Accuracy: {balanced_accuracy(y_test, predictions)}")