import bioscience as bs
###################
# 1) Load dataset 
###################
print("Loading dataset...")
dataset = bs.loadNetwork(path="C:/Users/pablo/OneDrive/Escritorio/Bioinformática/bioscience/datasets/network2.csv", separator = ",", index_nodeA = 1, index_nodeB = 3, index_weight = 9, skipr = 0, head = 0, columnsRelatedWeight = [2, 4, 8])
print("Dataset loaded successfully.")
###################
# 2) Checking that everything is saved correctly
###################
for network in dataset.results:
    print(network)