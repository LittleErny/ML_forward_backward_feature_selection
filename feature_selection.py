import numpy as np
from numpy import array_equal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

RANDOM_STATE = 42

# (a) Generate Predictors and Noise
rng = np.random.default_rng(RANDOM_STATE)
n = 100
X = rng.normal(size=n)
epsilon = rng.normal(size=n)


# (b) Generating the Target Variable
beta_0 = 12
beta_1 = -2.3
beta_2 = 3.4
beta_3 = -45

Y = beta_0 + beta_1 * X + beta_2 * X**2 + beta_3 * X**3 + epsilon

# Creating a list of features:
features = [X**i for i in range(1, 11)]


# Function for training and testing the model for some selected features
def train_and_test(x_data):
    # Bring the data to the table format
    x_data_stacked = np.column_stack(x_data)

    # Do train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_data_stacked, Y, test_size=0.3, random_state=RANDOM_STATE)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict & Score
    y_pred = model.predict(x_test)
    return r2_score(y_test, y_pred)


def forward_stepwise_selection(max_amount_of_features=5):
    x_selected = []  # X_dataset with selected features only
    x_selected_last_iter = None  # I have no idea why we actually need this variable, but it was on the slide
    c_rate_best = float("-inf")  # The best r2 rate we managed to achieve by now with current selection of features

    while len(x_selected) < max_amount_of_features:  # Until we achieve the max amount of allowed features

        x_selected_last_iter = x_selected.copy()  # Again, no idea why we need this
        c_rate = [float("-inf")] * len(features)  # The r2 rates of the model, if we add j-th feature to the current feature set

        for j, xj in enumerate(features):  # Iterate through the features
            if not any(array_equal(xj, selected) for selected in x_selected):  # if this feature was not used yet

                # Test what happens if we add this feature
                x_selected_temp = x_selected + [xj]
                c_rate[j] = train_and_test(x_selected_temp)

        # Check which feature was the best to add
        x_best_addition = c_rate.index(max(c_rate))

        # If adding the best feature makes the result worse, break the cycle
        if c_rate[x_best_addition] > c_rate_best:
            x_selected.append(features[x_best_addition])
            c_rate_best = c_rate[x_best_addition]
        else:
            break

    return x_selected, x_selected_last_iter


def backward_stepwise_selection(min_amount_of_features=3):
    x_selected = features.copy()  # X_dataset with selected features only
    x_selected_last_iter = None  # I have no idea why we actually need this variable, but it was on the slide
    c_rate_best = float("-inf")  # The best r2 rate we managed to achieve by now with current selection of features

    while len(x_selected) > min_amount_of_features:  # Until we achieve the max amount of allowed features

        x_selected_last_iter = x_selected.copy()  # Again, no idea why we need this
        c_rate = [float("-inf")] * len(features)  # The r2 rates of the model, if we add j-th feature to the current feature set

        for j, xj in enumerate(features):  # Iterate through the features
            if any(array_equal(xj, selected) for selected in x_selected):  # if this feature was already used

                # Test what happens if we remove this feature
                x_selected_temp = x_selected.copy()
                x_selected_temp.remove(features[xj])

                c_rate[j] = train_and_test(x_selected_temp)

        # Check which feature was the best to remove
        x_best_substraction = c_rate.index(max(c_rate))

        # If removing the best feature makes the result worse, break the cycle
        if c_rate[x_best_substraction] > c_rate_best:
            x_selected.remove(features[x_best_substraction])
            c_rate_best = c_rate[x_best_substraction]
        else:
            break

    return x_selected, x_selected_last_iter

x_selected, x_selected_last_iter = forward_stepwise_selection()

x_selected_indices = [j + 1 for j, feature in enumerate(features) if
                      any(array_equal(feature, selected) for selected in x_selected)]
print(f"Selected feature indices: {x_selected_indices}")
print(f"Model R2 score: {train_and_test(x_selected)}")

print(len(x_selected))
print(train_and_test(x_selected))

# Testing the train_and_test function using the first 4 features
first_4_features = features[:4]
print(f"R2 score using the first 4 features: {train_and_test(first_4_features)}")


