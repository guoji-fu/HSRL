import numpy as np

def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {} 
    while True:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num

    return vectors

def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while True:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()

    return X, Y
    