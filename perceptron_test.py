#########################################
#    Experiment using dataset and.dat   #
#########################################

# import packages
import pandas as pd
from scripts.perceptron import PerceptronAlgorithm

# import dataset
dataset = pd.read_csv("scripts/and.dat", sep =" ", header = None)

# split independent and dependent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# training
model = PerceptronAlgorithm(eta = 0.1, max_epochs = 100, threshold = 1e-4)
model.fit(x, y)

print("Loss = ", model.loss)
print("Cost = ", model.cost_)

# make predictions
print("0 AND 0 = ", model.predict([0, 0]))
print("0 AND 1 = ", model.predict([0, 1]))
print("1 AND 0 = ", model.predict([1, 0]))
print("1 AND 1 = ", model.predict([1, 1]))

# make test
model.test(x, y)
print("Accuracy = ", model.accuracy, "%")
