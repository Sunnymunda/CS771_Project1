import numpy as np
from sklearn.svm import LinearSVC

def get_x_vector(row):
    row_d = 1 - 2 * row
    x_vector = np.cumprod(row_d)
    return x_vector

def get_final_vector(x_vector):
    outer_product = np.outer(x_vector, x_vector)
    # Get the indices of the upper triangle without including the diagonal
    upper_triangle_indices = np.triu_indices(len(x_vector), k=1)
    # Extract the upper triangle excluding the diagonal from the outer product
    final_vector = outer_product[upper_triangle_indices]
    return np.concatenate([x_vector, final_vector])

def my_map(X):
    # Apply get_x_vector to each row of X
    X_new = np.apply_along_axis(get_x_vector, axis=1, arr=X)
    # Apply get_final_vector to each element of x_vector
    X_new = np.apply_along_axis(get_final_vector, axis=1, arr=X_new)
    return X_new

def my_fit(train_data):
    """
    Learns two linear models W0, b0, W1, b1 using the training data.

    Parameters:
    train_data (np.array): Training data with each row containing 34 elements:
                           32-bit challenge and 2-bit response.

    Returns:
    tuple: W0, b0, W1, b1 (weights and bias terms for Response0 and Response1 models)
    """
    # Separate features (challenges) and labels (responses)
    X_train = train_data[:, :-2]  # 32-bit challenges
    y_train_0 = train_data[:, -2]  # Response0
    y_train_1 = train_data[:, -1]  # Response1

    # Map challenges to feature vectors
    X_train_mapped = my_map(X_train)

    # Train LinearSVC model for Response0
    model_0 = LinearSVC(dual=False, C=3, tol=0.00015)
    model_0.fit(X_train_mapped, y_train_0)

    # Train LinearSVC model for Response1
    model_1 = LinearSVC(dual=False, C=3, tol=0.00015)
    model_1.fit(X_train_mapped, y_train_1)

    # Extract weights and bias terms
    W0 = model_0.coef_[0]
    b0 = model_0.intercept_[0]
    W1 = model_1.coef_[0]
    b1 = model_1.intercept_[0]

    return W0, b0, W1, b1

def predict_responses(challenges, W0, b0, W1, b1):
    """
    Predicts Response0 and Response1 for given challenges using the learned linear models.

    Parameters:
    challenges (np.array): Array of challenges to predict responses for.
    W0, b0: Parameters for the Response0 model.
    W1, b1: Parameters for the Response1 model.

    Returns:
    tuple: Predicted Response0 and Response1
    """
    # Map challenges to feature vectors
    X_mapped = my_map(challenges)

    # Predict Response0
    response_0 = (1 + np.sign(np.dot(X_mapped, W0) + b0)) // 2

    # Predict Response1
    response_1 = (1 + np.sign(np.dot(X_mapped, W1) + b1)) // 2

    return response_0, response_1

if __name__ == "__main__":
    # Load the training data
    train_data = np.loadtxt('public_trn.txt')

    # Learn the linear models
    W0, b0, W1, b1 = my_fit(train_data)

    # Load the test data
    test_data = np.loadtxt('public_tst.txt')
    X_test = test_data[:, :-2]  # 32-bit challenges
    y_test_0 = test_data[:, -2]  # Response0
    y_test_1 = test_data[:, -1]  # Response1

    # Predict responses
    pred_0, pred_1 = predict_responses(X_test, W0, b0, W1, b1)

    # Evaluate the model
    accuracy_0 = np.mean(pred_0 == y_test_0)
    accuracy_1 = np.mean(pred_1 == y_test_1)

    print(f'Accuracy for Response0: {accuracy_0 * 100:.2f}%')
    print(f'Accuracy for Response1: {accuracy_1 * 100:.2f}%')