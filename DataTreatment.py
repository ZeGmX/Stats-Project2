import numpy as np
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler


file = open(r"D:\Téléchargements\ProjetStats2-master\spambase.data", 'r')
spam_data = []
line = file.readline()
while line != "":
    entry = (line.rstrip()).split(',')
    spam_data.append(entry)
    line = file.readline()


spam_data = np.array(spam_data, dtype=np.double)
#print(spam_data[0], spam_data[1])

test = NeuralNetwork([2,2], 3)

outputs = []
for k in range(100):
    if spam_data[k, -1] == 1:
        outputs.append([0, 1])
    else:
        outputs.append([1, 0])

scaler = StandardScaler()
spam_data = scaler.fit_transform(spam_data)


#print(spam_data[:100,:3])

test.compute_all(spam_data[:100,:3], outputs)
test.update_coeff()
print(test.total_error())

test.compute_all(spam_data[:100,:3], outputs)
test.update_coeff()
print(test.total_error())