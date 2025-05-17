import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

RANDOM_STATE = 42

def generate_data():
    # Generate Predictors and Noise
    rng = np.random.default_rng(RANDOM_STATE)
    n = 100
    X = rng.normal(size=n)
    epsilon = rng.normal(size=n)

    # Generating the Target Variable
    beta_0 = 12
    beta_1 = -2.3
    beta_2 = 3.4
    beta_3 = -45

    Y = beta_0 + beta_1 * X + beta_2 * X**2 + beta_3 * X**3 + epsilon

    # Creating a list of features:
    list_of_features = [X**i for i in range(1, 11)]

    return X, Y, list_of_features


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


def forward_stepwise_selection(max_amount_of_features=5, x_selected_indices=None):
    if x_selected_indices is None:
        x_selected_indices = set()  # the set of indices of selected features
    all_features_indices = set(range(len(init_features)))  # set with all feature numbers
    x_selected = []  # the set of selected features
    x_left_indices = all_features_indices.copy()

    c_rate_best = float("-inf")  # The best r2 rate we managed to achieve by now with current selection of features

    while len(x_selected_indices) < max_amount_of_features:  # Until we achieve the max amount of allowed features

        c_rate = {feature_num: float("-inf") for feature_num in x_left_indices}
        for j in x_left_indices:  # Iterate through the not used features

            # Test what happens if we add this feature
            feature = init_features[j]
            x_selected_temp = x_selected.copy()
            x_selected_temp.append(feature)
            c_rate[j] = train_and_test(x_selected_temp)

        # Check which feature was the best to add
        c_rate_list = c_rate.items()
        x_best_addition = sorted(c_rate_list, key=lambda x: x[1], reverse=True)[0]

        # If adding the best feature makes the result worse, break the cycle
        if x_best_addition[1] > c_rate_best:
            x_selected_indices.add(x_best_addition[0])
            x_selected.append(init_features[x_best_addition[0]])
            c_rate_best = x_best_addition[1]
        else:
            break

    return x_selected, x_selected_indices


def backward_stepwise_selection(min_amount_of_features=3, x_included_indices=None):
    if x_included_indices is None:
        x_included_indices = set(range(len(init_features)))
    x_selected_indices = x_included_indices.copy()  # Start with all features
    x_selected = [init_features[i] for i in x_selected_indices]

    c_rate_best = float("-inf")  # The best r2 rate we managed to achieve by now with current selection of features

    while len(x_selected_indices) > min_amount_of_features:  # Until we achieve the min amount of allowed features

        c_rate = {feature_num: float("-inf") for feature_num in x_selected_indices}
        for j in x_selected_indices:  # Iterate through the currently included features

            # Test what happens if we remove this feature
            x_selected_temp = x_selected.copy()
            x_selected_temp = [feature for feature in x_selected_temp if not np.array_equal(feature, init_features[j])]
            c_rate[j] = train_and_test(x_selected_temp)

        # Check which feature is the best to remove
        c_rate_list = c_rate.items()
        x_best_removal = sorted(c_rate_list, key=lambda x: x[1], reverse=True)[0]

        # If removing the best feature makes the result better, update the selection
        if x_best_removal[1] > c_rate_best:
            x_selected_indices.remove(x_best_removal[0])
            x_selected = [init_features[i] for i in x_selected_indices]
            c_rate_best = x_best_removal[1]
        else:
            break

    return x_selected, x_selected_indices


# Generate some data & features
X, Y, init_features = generate_data()

# Try only forward-stepwise-selection(FSS)
x_selected, x_selected_indices = forward_stepwise_selection()
print("Only forward stepwise selection:")
print(f"Selected feature indices: {x_selected_indices}")
print(f"Model R2 score: {train_and_test(x_selected)}")
print("\n- - -\n")

# Try only backward-stepwise-selection(BSS)
x_selected, x_selected_indices = backward_stepwise_selection()
print("Only backward stepwise selection:")
print(f"Selected feature indices: {x_selected_indices}")
print(f"Model R2 score: {train_and_test(x_selected)}")
print("\n- - -\n")

# Try FSS + BSS
print("Both stepwise selection(forward and backward):")
x_selected, x_selected_indices = forward_stepwise_selection(max_amount_of_features=10)
x_selected, x_selected_indices = backward_stepwise_selection(min_amount_of_features=3, x_included_indices=x_selected_indices)
print(f"Selected feature indices: {x_selected_indices}")
print(f"Model R2 score: {train_and_test(x_selected)}")
print("\n- - -\n")

# Try BSS + FSS
print("Both stepwise selection(backward and forward):")
x_selected, x_selected_indices = backward_stepwise_selection(min_amount_of_features=3)
x_selected, x_selected_indices = forward_stepwise_selection(max_amount_of_features=10, x_selected_indices=x_selected_indices)
print(f"Selected feature indices: {x_selected_indices}")
print(f"Model R2 score: {train_and_test(x_selected)}")
