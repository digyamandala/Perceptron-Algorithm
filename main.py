import numpy as np
import csv
import random
from random import choice

n_input = 3
n_output = 1
epochs = 500
learning_rate = 0.1

weights = np.random.normal(size=(n_input, n_output))
biases = np.random.normal(size=(n_output))


def binaryFunction(x):
    if(x >= 0):
        return 1
    return 0

def loadData(filepath):
    with open(filepath) as f:
        readline = csv.reader(f)
        next(readline)
        dataset = []
        for i in readline:
            features = i[:3]
            labels = i[3]
            dataset.append([features, labels])
        return dataset

def preprocess(dataset):
    preprodata= []

    for i,j in dataset:
        feature = [float(x) for x in i]

        if(j == "Lying"):
            label = 1
        else:
            label = 0
        preprodata.append([np.array(feature),np.array(label)])
    return preprodata

def forwardPass(x):
    act = np.matmul(x,weights) + biases
    act = binaryFunction(act)
    return act

train_res = []
def train(dataset):
    for i in range(0, epochs):
        x,t = choice(dataset)
        y = forwardPass(x)
        err = t - y

        #update w
        global weights
        weights = weights + (learning_rate * err  * x.reshape(n_input,n_output))

        #update b
        global biases
        biases = biases + (learning_rate * err)

        cp = 0
        for data in dataset:
            res = forwardPass(data[0])
            if(res == data[1]):
                cp = cp + 1

        acc = (cp/len(dataset)) * 100
        if(i % 50 == 0):
            # print(f"epoch {i}: {acc}%")
            train_res.append([i, acc])


def test(datax):
    y = forwardPass(datax)
    return y
    

def menu():
    print("1. Detect Lie")
    print("2. Train")
    print("3. Exit")
    print(">> ", end="")

def cls():
    for i in range(0,30):
        print()

def main():
    dataset = loadData("lie-dataset.csv")
    dataset = preprocess(dataset)
    random.shuffle(dataset)
    train(dataset)
    c = -1
    while(c != 3):
        cls()
        menu()
        c = int(input())
        if(c == 1):
            heart = input("Heart rate [1-10]: ")
            blink = input("Blink rate [1-10]: ")
            eyecont = input("Eye Contact rate [1-10]: ")
            res = test([int(heart),int(blink),int(eyecont)])
            if(res == 1):
                print("She/He tells lies")
            else:
                print("She/He tells truth")
            input("Press Enter")

        elif(c == 2):
            for i,j in train_res:
                print(f"Accuracy in Epoch {i}: {j}%")
            input("Press Enter")
                
main()
