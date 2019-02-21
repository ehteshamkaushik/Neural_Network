import random
import math
import numpy as np

a = 1
u = .1


def get_train(fname):
    filename = fname
    with open(filename) as fp:
        line = fp.readline().split()
        x = int(line[0])
        n = int(line[1])
        l = int(line[2])
        attr = []
        label = []
        value = 0
        for line in fp:
            temp = line.split()
            temp_list = []
            for i in range(x):
                if i == 0:
                    value+= float(temp[i])
                temp_list.append(float(temp[i]))
            temp_list.append(1)
            attr.append(temp_list)
            label.append(int(temp[x]))
    return n, x, l, attr, label


def get_test(fname):
    filename = fname
    with open(filename) as fp:
        attr = []
        label = []
        for line in fp:
            temp = line.split()
            temp_list = []
            for i in range(x):
                temp_list.append(float(temp[i]))
            temp_list.append(1)
            attr.append(temp_list)
            label.append(int(temp[x]))
    return attr, label


def normalize(features):
    mean = np.asarray(features).mean(axis=0)
    variance = np.asarray(features).std(axis=0)
    #print(mean)
    #print(variance)
    for i in range(l):
        for j in range(x):
            features[i][j] = ((features[i][j] - mean[j])/variance[j])


def get_sigmoid(v):
    global a
    sigmoid = math.exp(-a*v)
    sigmoid += 1
    sigmoid = 1/sigmoid
    return sigmoid


def get_sigmoid_prime(v):
    sigmoid_prime = get_sigmoid(v)*(1 - get_sigmoid(v))
    return sigmoid_prime


class Perceptron:

    def __init__(self, dim):
        self.x_in = []
        self.dim = dim
        self.v = 0.0
        self.weight = [random.uniform(0, 1) for j in range(self.dim)]
        self.delta = 0.0
        self.out = 0.0
        self.error = 0.0

    def get_v(self):
        #print(self.weight)
        self.v = np.matmul(np.asarray(self.x_in).transpose(), np.asarray(self.weight))

    def get_out(self):
        self.out = get_sigmoid(self.v)

    def get_delta(self):
        #print(self.v)
        #print(self.error)
        self.delta = self.error * get_sigmoid_prime(self.v)

    def update_weight(self):
        global u
        self.weight = np.asarray(self.weight) - u*self.delta*np.asarray(self.x_in)


class Layer:
    def __init__(self, no_of_perceptron, dim, is_output_layer):
        self.no_of_perceptron = no_of_perceptron
        self.is_output_layer = is_output_layer
        self.dim = dim
        self.perceptrons = [Perceptron(self.dim) for i in range(self.no_of_perceptron)]
        self.x_in = []
        self.label = -1
        self.x_out = []

    def set_current_sample(self, x_in, label):
        self.x_in = x_in
        self.label = label
        for i in range(self.no_of_perceptron):
            self.perceptrons[i].x_in = self.x_in

    def calculate_error(self):
        for i in range(self.no_of_perceptron):
            if i == self.label - 1:
                self.perceptrons[i].error = self.perceptrons[i].out - 1
            else:
                self.perceptrons[i].error = self.perceptrons[i].out
            #print("Error : ", self.perceptrons[i].error)

    def set_error(self, error):
        for i in range(self.no_of_perceptron):
            #print("Perceptron : ", i)
            self.perceptrons[i].error = error[i]
            #print("Error : ", self.perceptrons[i].error)

    def calculate_previous_error(self):
        error = []
        for i in range(self.dim):
            val = 0
            for j in range(self.no_of_perceptron):
                val += (self.perceptrons[j].weight[i] * self.perceptrons[j].delta)
            error.append(val)
        return error

    def prepare_perceptron(self):
        #print("Previous out : ", self.x_out)
        self.x_out.clear()
        #print("Cleared out", self.x_out)
        for i in range(self.no_of_perceptron):
            #print("----------Perceptron : ", i)
            self.perceptrons[i].get_v()
            #print("V : ", self.perceptrons[i].v)
            self.perceptrons[i].get_out()
            #print("Out : ", self.perceptrons[i].out)
            self.x_out.append(self.perceptrons[i].out)
        self.x_out.append(1)
        #print("Output from layer : ", self.x_out)

    def update_preceptron(self):
        #print("Updating Perceptron : ")
        for i in range(self.no_of_perceptron):
            #print("Perceptron : ", i)
            self.perceptrons[i].get_delta()
            #print("Delta : ", self.perceptrons[i].delta)
            #print("Previous Weight : ", self.perceptrons[i].weight)
            self.perceptrons[i].update_weight()
            #print("Updated Weight : ", self.perceptrons[i].weight)

    def get_result(self):
        idx = -1
        val = -100000
        for i in range(self.no_of_perceptron):
            #print(i)
            #print(self.perceptrons[i].out)
            #x = input()
            if val < self.perceptrons[i].out:
                val = self.perceptrons[i].out
                idx = i
        #print(idx)
        #print(self.label - 1)
        if idx == self.label - 1:
            return 1
        else:
            return 0


n, x, l, training_attributes, training_labels = get_train("Train.txt")
test_attributes, test_labels = get_test("Test.txt")
normalize(training_attributes)
normalize(test_attributes)

#print("Train ", training_attributes)

no_of_inter_layers = 2
no_of_perceptrons = [5, 4]

layers = []
layers.append(Layer(x, x+1, False))
for i in range(no_of_inter_layers):
    layers.append(Layer(no_of_perceptrons[i], layers[i].no_of_perceptron + 1, False))

layers.append(Layer(n, layers[no_of_inter_layers].no_of_perceptron + 1, True))


def testData(data, label):
    #print(data)
    true = 0
    for i in range(len(data)):
        x_in = data[i]
        x_label = label[i]

        for j in range(len(layers)):
            layers[j].set_current_sample(x_in, x_label)
            layers[j].prepare_perceptron()
            if layers[j].is_output_layer:
                true += layers[j].get_result()
            else:
                x_in = layers[j].x_out
    accuracy = (true / len(training_attributes)) * 100
    missed = l - true
    print("Misclassified : ", missed)
    #print("Accuracy", accuracy)
    return accuracy


for k in range(500):
    print("Run : ", k)
    accuracy = testData(test_attributes, test_labels)
    print("Accuracy", accuracy)
    if accuracy == 100:
        break
    #t = input()
    for i in range(len(training_attributes)):
        #print("Starting Train : ")
        x_in = training_attributes[i]
        #print("Training Data : ", x_in)
        x_label = training_labels[i]
        #print("Training Lable : ", x_label)
        #print("---------------Forward -------------")
        for j in range(len(layers)):
            #print("At layer : ", j)
            layers[j].set_current_sample(x_in, x_label)
            #print("Sample Found in Layer : ", layers[j].x_in, layers[j].label)
            #print("Preparing PT : ")
            layers[j].prepare_perceptron()
            if layers[j].is_output_layer:
                #print("Final Layer : ")
                layers[j].calculate_error()
            else:
                #print("Not Final Layer : ")
                x_in = layers[j].x_out
                #print("Updated Value of Input : ", x_in)
        error = []
        #t = input()
        #print("##################################Backward : ")
        for j in range(len(layers)-1, -1, -1):
            #print("Layer : ", j)
            if layers[j].is_output_layer:
                #print("Final Layer : ")
                layers[j].update_preceptron()
                error = layers[j].calculate_previous_error()
                #print("Previous Error : ", error)
            else:
                #print("Not Final Layer : ")
                #print("Setting Error : ")
                layers[j].set_error(error)
                layers[j].update_preceptron()
                error = layers[j].calculate_previous_error()
                #print("Previous Error : ", error)
'''
testData(test_attributes, test_labels)




true = 0
for i in range(len(test_attributes)):
    x_in = test_attributes[i]
    x_label = test_labels[i]

    for j in range(len(layers)):
        layers[j].set_current_sample(x_in, x_label)
        layers[j].prepare_perceptron()
        if layers[j].is_output_layer:
            true += layers[j].get_result()
        else:
            x_in = layers[j].x_out


accuracy = (true/len(training_attributes))*100
print("Accuracy", accuracy)
'''


