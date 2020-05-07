import numpy as np
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def set_to_True_False(prediction):
    N = len(prediction)
    res = np.zeros(N, dtype=np.bool)
    for k in range(N):
        line = prediction[k]
        if np.abs(line[0]) <= np.abs(line[1]):
            res[k] = True
    return res


def full_test(train_size=100, number_of_columns=57, n=20, format=[5, 3, 2]):
    sample_rows = np.random.choice(len(spam_data), train_size, replace=False)
    sample_columns = np.random.choice(58, number_of_columns, replace=False)
    inputs_train = spam_data[sample_rows][:, sample_columns]
    outputs_train = outputs[sample_rows]

    nn = NeuralNetwork(format, number_of_columns)
    y_list = nn.train(inputs_train, outputs_train, n)

    t = np.ones(len(spam_data), dtype=np.bool)
    t[sample_rows] = False
    inputs_testing = spam_data[t][:, sample_columns]

    pred = nn.predict(inputs_testing)
    outputs_testing = set_to_True_False(pred)
    expected_outputs = spam_data[t, -1] > 0
    match = outputs_testing == expected_outputs
    print("Success rate : {}%".format(sum(match) / len(match)))

    fig = plt.figure()
    fig.suptitle("Evolution de l'erreur")
    plt.plot(y_list)
    plt.xlabel("Nombre de passages dans le réseau")
    plt.ylabel("Erreur totale")
    plt.show()


if __name__ == "__main__":
    file = open(r"D:\Téléchargements\ProjetStats2-master\spambase.data", 'r')
    spam_data = []
    line = file.readline()
    while line != "":
        entry = (line.rstrip()).split(',')
        spam_data.append(entry)
        line = file.readline()


    spam_data = np.array(spam_data, dtype="float64")
    scaler = StandardScaler()
    spam_data = scaler.fit_transform(spam_data)

    outputs = np.zeros((len(spam_data), 2), dtype=np.double)
    for k in range(len(spam_data)):
        if spam_data[k, -1] > 0:
            outputs[k] = [0, 1]
        else:
            outputs[k] = [1, 0]


    full_test(100, 10, 50)
