# Implemented by Mikhail Avrutskii for the ML Course at THD; 18.05.2025

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Some constants
RANDOM_STATE = 131

# The length of dataset
N = 100
NUMBER_OF_FEATURES = 10
TEST_SIZE = 0.3

# Some params for generating target variable
BETA_0 = 20
BETA_1 = -5
BETA_2 = 4
BETA_3 = -1


# Function to generate fake data
def generate_data():
    # Generate Predictors and Noise
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=N)
    epsilon = rng.normal(size=N)

    # Generating the Target Variable
    Y = BETA_0 + BETA_1 * X + BETA_2 * X ** 2 + BETA_3 * X ** 3 + epsilon

    # Creating a list of features:
    list_of_features = [X ** i for i in range(1, NUMBER_OF_FEATURES + 1)]

    return X, Y, list_of_features


def calculate_sigma2(x_data):
    x_data_stacked = np.column_stack(x_data)

    # Do train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_data_stacked, Y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)

    residuals = y_test - y_pred
    sigma2 = sum(residuals ** 2) / (N - NUMBER_OF_FEATURES - 1)  # - 1 because of intercept

    return sigma2


def calculate_Cp(rss, d):
    # The formula was taken from the ISLP book (formula 6.2, page 236)
    return 1 / N * (rss + 2 * d * SIGMA_SQUARED)


# Function for training and testing the model for some selected features
def train_and_test(x_data):
    # Bring the data to the table format
    x_data_stacked = np.column_stack(x_data)

    # Do train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_data_stacked, Y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict & Score
    y_pred = model.predict(x_test)
    # Calculate Sum of Squared Errors (SSE), which is exactly the Residual Sum of Squares (RSS)
    rss = mean_squared_error(y_test, y_pred) * N

    # Number of model parameters (parameters + intercept)
    p = len(x_data) + 1

    # Возвращаем C_p, r2_score, and the values of model coefficients
    return calculate_Cp(rss, p), r2_score(y_test, y_pred), model.intercept_, model.coef_


def forward_stepwise_selection(max_amount_of_features=7, included_feature_indices=None):
    if included_feature_indices is None:
        included_feature_indices = set()  # the set of indices of selected features
    all_features_indices = set(range(len(init_features)))  # set with all feature numbers
    selected_features = [init_features[i] for i in included_feature_indices]  # the list of selected features
    left_feature_indices = all_features_indices - included_feature_indices

    best_cp_score = train_and_test(
        selected_features)[0] if included_feature_indices else float(
        "inf")  # The best Cp score we managed to achieve by now with current selection of features

    while len(included_feature_indices) < max_amount_of_features:  # Until we achieve the max amount of allowed features

        cp_scores = {feature_num: float("inf") for feature_num in left_feature_indices}
        for j in left_feature_indices:  # Iterate through the not used features

            # Test what happens if we add this feature
            feature = init_features[j]
            selected_features_temp = [init_features[i] for i in included_feature_indices | {j}]
            cp_scores[j] = train_and_test(selected_features_temp)[0]

        # Check which feature was the best to add
        cp_scores_list = cp_scores.items()
        best_feature_to_add = min(cp_scores_list, key=lambda x: x[1])

        # If adding the best feature makes the result worse, break the cycle
        if best_feature_to_add[1] < best_cp_score:
            left_feature_indices.remove(best_feature_to_add[0])
            included_feature_indices.add(best_feature_to_add[0])
            selected_features.append(init_features[best_feature_to_add[0]])
            best_cp_score = best_feature_to_add[1]
        else:
            break

    return included_feature_indices


def backward_stepwise_selection(min_amount_of_features=3, included_feature_indices=None):
    if included_feature_indices is None:
        included_feature_indices = set(range(NUMBER_OF_FEATURES))  # just use all the indices we have
    selected_features = [init_features[i] for i in included_feature_indices]

    best_cp_score = train_and_test(
        selected_features)[0]  # The best Cp score we managed to achieve by now with current selection of features

    while len(included_feature_indices) > min_amount_of_features:  # Until we achieve the min amount of allowed features

        cp_scores = {feature_num: float("inf") for feature_num in included_feature_indices}
        for j in included_feature_indices:  # Iterate through the currently included features

            # Test what happens if we remove this feature
            selected_features_temp = selected_features.copy()
            selected_features_temp = [feature for feature in selected_features_temp if
                                      not np.array_equal(feature, init_features[j])]
            cp_scores[j] = train_and_test(selected_features_temp)[0]

        # Check which feature is the best to remove
        cp_scores_list = cp_scores.items()
        best_feature_to_remove = min(cp_scores_list, key=lambda x: x[1])

        # If removing the best feature makes the result better, update the selection
        if best_feature_to_remove[1] < best_cp_score:
            included_feature_indices.remove(best_feature_to_remove[0])
            selected_features = [init_features[i] for i in included_feature_indices]
            best_cp_score = best_feature_to_remove[1]
        else:
            break

    return included_feature_indices


# Rearranges polynomial coefficients to align with the full set of features, marking excluded features with None.
def arrange_polynomial_coefficients(coefs, included_feature_indices):
    predicted_polynomial_coeficients = []
    coefs_index = 0

    for i in range(NUMBER_OF_FEATURES):
        if i in included_feature_indices:
            predicted_polynomial_coeficients.append(coefs[coefs_index])
            coefs_index += 1
        else:
            predicted_polynomial_coeficients.append(None)

    return predicted_polynomial_coeficients


# Just a func for nice printing a table
def print_coefs_table(predicted_polynomial_coeficients):
    print(f'Predicted polynomial coeficients:')
    print("Polynomial Degree", end='\t|\t\t')
    [print(i, end='\t\t|\t\t') for i in range(NUMBER_OF_FEATURES + 1)]
    print()
    print("--------------------", end='+-------')
    [print("--------+-------", end='') for i in range(NUMBER_OF_FEATURES + 1)]
    print()
    print("Coefficient", end='\t\t\t|\t\t')
    [print(f'{coef:.2f}' if coef is not None else "None", end='\t|\t\t') for coef in predicted_polynomial_coeficients]
    print()


# ------------------ Start of Execution ------------------

# Generate some data & features
X, Y, init_features = generate_data()

SIGMA_SQUARED = calculate_sigma2(init_features)

# Try only forward-stepwise-selection(FSS)
included_feature_indices = forward_stepwise_selection()
model_cp, r2, intercept, coefs = train_and_test([init_features[i] for i in sorted(list(included_feature_indices))])
predicted_polynomial_coeficients = arrange_polynomial_coefficients(coefs, included_feature_indices)

print("Only forward stepwise selection:")
print(f"Model R2 score: {r2}")
print(f"Model Cp score: {model_cp:.2f}")
print_coefs_table([intercept] + predicted_polynomial_coeficients)

print("\n- - -\n")

# Try only backward-stepwise-selection(BSS)
included_feature_indices = backward_stepwise_selection()
model_cp, r2, intercept, coefs = train_and_test([init_features[i] for i in sorted(list(included_feature_indices))])
predicted_polynomial_coeficients = arrange_polynomial_coefficients(coefs, included_feature_indices)

print("Only backward stepwise selection:")
print(f"Model R2 score: {r2}")
print(f"Model Cp score: {model_cp:.2f}")
print_coefs_table([intercept] + predicted_polynomial_coeficients)

print("\n- - -\n")

# Try FSS + BSS
print("Both stepwise selection (forward and backward):")
included_feature_indices = forward_stepwise_selection(max_amount_of_features=10)
included_feature_indices = backward_stepwise_selection(min_amount_of_features=3,
                                                       included_feature_indices=included_feature_indices)
model_cp, r2, intercept, coefs = train_and_test([init_features[i] for i in sorted(list(included_feature_indices))])
predicted_polynomial_coeficients = arrange_polynomial_coefficients(coefs, included_feature_indices)

print(f"Model R2 score: {r2}")
print(f"Model Cp score: {model_cp:.2f}")
print_coefs_table([intercept] + predicted_polynomial_coeficients)

print("\n- - -\n")

# Try BSS + FSS
print("Both stepwise selection (backward and forward):")
included_feature_indices = backward_stepwise_selection(min_amount_of_features=3)
included_feature_indices = forward_stepwise_selection(max_amount_of_features=10,
                                                      included_feature_indices=included_feature_indices)
model_cp, r2, intercept, coefs = train_and_test([init_features[i] for i in sorted(list(included_feature_indices))])
predicted_polynomial_coeficients = arrange_polynomial_coefficients(coefs, included_feature_indices)

print(f"Model R2 score: {r2}")
print(f"Model Cp score: {model_cp:.2f}")
print_coefs_table([intercept] + predicted_polynomial_coeficients)
