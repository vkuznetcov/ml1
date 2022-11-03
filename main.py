import numpy as np
import pandas as pd

raw_data_path = "datasets/data.txt"
raw_true_path = "datasets/data_true.txt"
raw_fake_path = "datasets/data_fake.txt"


def read_data(file, file_true, file_fake):
    return pd.read_csv(file, sep=':', index_col=0, names=["User", "Data"]) \
        , pd.read_csv(file_true, sep=':', index_col=0, names=["User", "Data"]) \
        , pd.read_csv(file_fake, sep=':', index_col=0, names=["User", "Data"])


def get_transit_matrix(data, states):
    matrix = np.full((len(states), len(states)), 0.1)

    for data_index in range(len(data) - 1):
        matrix[states.index(data[data_index]), states.index(data[data_index + 1])] += 1

    for index in range(len(states)):
        matrix[index] = matrix[index] / matrix[index].sum()

    return matrix


def get_states_set(data):
    result = []
    for user, user_data in data.iterrows():
        result += user_data.Data.split(";")
    return list(set(result))


def get_window_trend(data, matrix, states, prob=1):
    for index in range(len(data) - 1):
        prob *= matrix[states.index(data[index]), states.index(data[index + 1])]
    return prob


def get_trends(data, matrix, window, states):
    probs = []

    if len(data) > window:
        for i in range(len(data) - window):
            win_data = data[i: i + window]
            probs.append(get_window_trend(win_data, matrix, states))
    else:
        probs.append(get_window_trend(data, matrix, states))

    return probs


def anomalies_checking(data, matrix, window, interval, states):
    probs = get_trends(data, matrix, window, states)

    for prob in probs:
        if not (interval[0] < prob < interval[1]):
            return 1
    return 0


def calculate_interval(data, matrix, window, states):
    probs = get_trends(data, matrix, window, states)
    array = np.array(probs)
    return np.min(array), np.max(array)


if __name__ == '__main__':
    raw_data, raw_true, raw_fake = read_data(raw_data_path, raw_true_path, raw_fake_path)

    all_states = get_states_set(raw_data)
    window = 10
    result_true = []
    result_fake = []

    for user, user_data in raw_data.iterrows():
        data_list = user_data.Data.split(";")
        data_true_list = raw_true.Data[user].split(";")
        data_fake_list = raw_fake.Data[user].split(";")

        transit_matrix = get_transit_matrix(data_list, all_states)
        interval = calculate_interval(data_list, transit_matrix, window, all_states)

        result_true.append(anomalies_checking(data_true_list, transit_matrix, window, interval, all_states))
        result_fake.append(anomalies_checking(data_fake_list, transit_matrix, window, interval, all_states))

    print("Data_true: ", sum(result_true) / raw_true.shape[0])
    print("Data_fake: ", 1 - sum(result_fake) / raw_fake.shape[0])
