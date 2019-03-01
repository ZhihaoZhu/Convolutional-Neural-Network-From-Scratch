import pickle

with open("../data/sampledCIFAR10" , "rb") as f :
    data = pickle.load(f)
    train, val, test = data["train"], data["val"], data["test"]
    print("loaded success")