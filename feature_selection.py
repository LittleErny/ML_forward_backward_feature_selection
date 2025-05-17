import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Some constants
RANDOM_STATE = 131
BETA_0 = 20
BETA_1 = -10
BETA_2 = 4
BETA_3 = -1


# Function to generate fake data
def generate_data():
    # Generate Predictors and Noise
    rng = np.random.default_rng(RANDOM_STATE)
    n = 100
    X = rng.normal(size=n)
    epsilon = rng.normal(size=n)

    # Generating the Target Variable
    Y = BETA_0 + BETA_1 * X + BETA_2 * X ** 2 + BETA_3 * X ** 3 + epsilon

    # Creating a list of features:
    list_of_features = [X ** i for i in range(1, 11)]

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


def forward_stepwise_selection(max_amount_of_features=5, included_feature_indices=None):
    if included_feature_indices is None:
        included_feature_indices = set()  # the set of indices of selected features
    all_features_indices = set(range(len(init_features)))  # set with all feature numbers
    selected_features = [init_features[i] for i in included_feature_indices]  # the list of selected features
    left_feature_indices = all_features_indices.copy()

    best_r2_score = float("-inf")  # The best r2 rate we managed to achieve by now with current selection of features

    while len(included_feature_indices) < max_amount_of_features:  # Until we achieve the max amount of allowed features

        r2_scores = {feature_num: float("-inf") for feature_num in left_feature_indices}
        for j in left_feature_indices:  # Iterate through the not used features

            # Test what happens if we add this feature
            feature = init_features[j]
            selected_features_temp = selected_features.copy()
            selected_features_temp.append(feature)
            r2_scores[j] = train_and_test(selected_features_temp)

        # Check which feature was the best to add
        r2_scores_list = r2_scores.items()
        best_feature_to_add = sorted(r2_scores_list, key=lambda x: x[1], reverse=True)[0]

        # If adding the best feature makes the result worse, break the cycle
        if best_feature_to_add[1] > best_r2_score:
            included_feature_indices.add(best_feature_to_add[0])
            selected_features.append(init_features[best_feature_to_add[0]])
            best_r2_score = best_feature_to_add[1]
        else:
            break

    return selected_features, included_feature_indices


def backward_stepwise_selection(min_amount_of_features=3, included_feature_indices=None):
    if included_feature_indices is None:
        included_feature_indices = set(range(len(init_features)))
    selected_features = [init_features[i] for i in included_feature_indices]

    best_r2_score = train_and_test(
        selected_features)  # The best r2 rate we managed to achieve by now with current selection of features

    while len(included_feature_indices) > min_amount_of_features:  # Until we achieve the min amount of allowed features

        r2_scores = {feature_num: float("-inf") for feature_num in included_feature_indices}
        for j in included_feature_indices:  # Iterate through the currently included features

            # Test what happens if we remove this feature
            selected_features_temp = selected_features.copy()
            selected_features_temp = [feature for feature in selected_features_temp if
                                      not np.array_equal(feature, init_features[j])]
            r2_scores[j] = train_and_test(selected_features_temp)

        # Check which feature is the best to remove
        r2_scores_list = r2_scores.items()
        best_feature_to_remove = sorted(r2_scores_list, key=lambda x: x[1], reverse=True)[0]

        # If removing the best feature makes the result better, update the selection
        if best_feature_to_remove[1] > best_r2_score:
            included_feature_indices.remove(best_feature_to_remove[0])
            selected_features = [init_features[i] for i in included_feature_indices]
            best_r2_score = best_feature_to_remove[1]
        else:
            break

    return selected_features, included_feature_indices


# Generate some data & features
X, Y, init_features = generate_data()

# Try only forward-stepwise-selection(FSS)
selected_features, included_feature_indices = forward_stepwise_selection()
print("Only forward stepwise selection:")
print(f"Selected feature indices: {included_feature_indices}")
print(f"Model R2 score: {train_and_test(selected_features)}")
print("\n- - -\n")

# Try only backward-stepwise-selection(BSS)
selected_features, included_feature_indices = backward_stepwise_selection()
print("Only backward stepwise selection:")
print(f"Selected feature indices: {included_feature_indices}")
print(f"Model R2 score: {train_and_test(selected_features)}")
print("\n- - -\n")

# Try FSS + BSS
print("Both stepwise selection(forward and backward):")
selected_features, included_feature_indices = forward_stepwise_selection(max_amount_of_features=10)
selected_features, included_feature_indices = backward_stepwise_selection(min_amount_of_features=3,
                                                                          included_feature_indices=included_feature_indices)
print(f"Selected feature indices: {included_feature_indices}")
print(f"Model R2 score: {train_and_test(selected_features)}")
print("\n- - -\n")

# Try BSS + FSS
print("Both stepwise selection(backward and forward):")
selected_features, included_feature_indices = backward_stepwise_selection(min_amount_of_features=3)
selected_features, included_feature_indices = forward_stepwise_selection(max_amount_of_features=10,
                                                                         included_feature_indices=included_feature_indices)
print(f"Selected feature indices: {included_feature_indices}")
print(f"Model R2 score: {train_and_test(selected_features)}")
