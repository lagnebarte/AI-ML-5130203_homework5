##########################################
#    Experiment using dataset xor.dat    #
##########################################

# import packages
import pandas as pd
from scripts.mlp import MLPAlgorithm

# import dataset
dataset = pd.read_csv("scripts/xor.dat", sep =" ", header = None) #самим найти датасет
# split independent and dependent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# build MLP architecture
model = MLPAlgorithm(eta = 0.1, threshold = 1e-4, max_epochs = 20000)
model.build_architecture(input_length = 2, hidden_length = 2, output_length = 1)         

# training step
model.fit(x, y)

# parameters of MLP architecture
print("Weights for hidden layers: ", model.Wh)
print("Bias for hidden layers: ", model.bh)
print("Weights for output layers: ", model.Wo)
print("Bias for output layers: ", model.bo)

# losses
print("Loss = ", model.loss_)

# output predictions
print("Output preds = ", model.fnet_o)

# make predictions
print("0 XOR 0 = ", model.predict([0, 0])[1])
print("0 XOR 1 = ", model.predict([0, 1])[1])
print("1 XOR 0 = ", model.predict([1, 0])[1])
print("1 XOR 1 = ", model.predict([1, 1])[1])

# testing
model.test(x, y)
print("Accuracy = ", model.accuracy, "%")
