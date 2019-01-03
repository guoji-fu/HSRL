import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from graph import *

import node2vec
import line

from hier_samp import louvainModularityOptimization
import matplotlib.pyplot as plt
import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input',                              
                        ## Link Prediction
                        # default='data/dblp/train_edges.txt',     
                        default='data/movielens/train_edges.txt',                   
                        # default='data/imdb/train_edges.txt',                                                                             
                        # default='data/douban/train_edges.txt',                        
                        # default='data/yelp/train_edges.txt',       
                        # default='data/mit/train_edges.txt', 

                        help='input graph file')
    parser.add_argument('--output',    
                        ## Link prediction                     
                        # default='output/dblp/dblp_dw_hs_lp_embeddings.txt',
                        # default='output/dblp/dblp_n2v_hs_lp_embeddings.txt',
                        # default='output/dblp/dblp_line_hs_lp_embeddings.txt',
                        
                        default='output/movielens/movielens_dw_hs_lp_embeddings.txt',
                        # default='output/movielens/movielens_n2v_hs_lp_embeddings.txt',
                        # default='output/movielens/movielens_line_hs_lp_embeddings.txt',                       
                        
                        # default='output/douban/douban_dw_hs_lp_embeddings.txt',                        
                        # default='output/douban/douban_n2v_hs_lp_embeddings.txt',                        
                        # default='output/douban/douban_line_hs_lp_embeddings.txt',                        
                        
                        # default='output/yelp/yelp_dw_hs_lp_embeddings.txt',                        
                        # default='output/yelp/yelp_n2v_hs_lp_embeddings.txt',                        
                        # default='output/yelp/yelp_line_hs_lp_embeddings.txt',                        
                        
                        # default='output/mit/mit_dw_hs_lp_embeddings.txt',                        
                        # default='output/mit/mit_n2v_hs_lp_embeddings.txt',                        
                        # default='output/mit/mit_line_hs_lp_embeddings.txt',

                        help='output representation file')
    parser.add_argument('--graph-format',
                        default='edgelist',
                        help='input graph format')
    parser.add_argument('--directed',
                        action='store_true',
                        help='treat graph as directed')
    parser.add_argument('--weighted',
                        action='store_true',
                        default=False,                        
                        help='treat graph as weighted')

    parser.add_argument('--representation-size',
                        default=16,
                        type=int,
                        help='number of latent dimensions to learn for each node')

    parser.add_argument('--method',
                        default='deepwalk',
                        # default='node2vec',                        
                        # default='line',                   
                        choices=['deepwalk', 'node2vec', 'line'],
                        help='the learning method')

    ## parameters for GraRep
    parser.add_argument('--Kstep',
                        default=4,
                        type=int,
                        help='use k-step transition probability matrix')
    
    ## parameters for deepwalk and note2vec
    parser.add_argument('--walk-length',
                        default=40,
                        type=int,
                        help='length of the random walk')
    parser.add_argument('--number-walks',
                        default=10,
                        type=int,
                        help='number of random walks to start at each node')
    parser.add_argument('--window-size',
                        default=5,
                        type=int,
                        help='window size of skipgram model')
    parser.add_argument('--workers',
                        default=10,
                        type=int,
                        help='number of parallel processes')
    parser.add_argument('--p',
                        default=1.0,
                        type=float)
    parser.add_argument('--q',
                        default=1.0,
                        type=float)

    ## parameters for LINE
    parser.add_argument('--order',
                        default=3,
                        type=int,
                        help='choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--epochs',
                        default=4,
                        type=int,
                        help='the training epochs of LINE or GCN')
    parser.add_argument('--no-auto-stop',
                        action='store_true',
                        default=False,
                        help='no early stop when training LINE')
    
    # parameters for Hierarchical Sampling
    parser.add_argument('--hs_num',
                        default=3,
                        type=int,
                        help='the number of hierarchical sampling layers')
    
    args = parser.parse_args()

    return args

def buildGraph(args):
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input,
                        weighted=args.weighted,
                        directed=args.directed)

    return g

def buildModel(args, g, batch_size=128, hs=0):
    if args.method == 'deepwalk':
        model = node2vec.Node2vec(graph=g,
                                    path_length=args.walk_length,
                                    num_paths=args.number_walks,
                                    dim=args.representation_size,
                                    workers=args.workers,
                                    p=args.p,
                                    q=args.q,
                                    dw=True,
                                    window=args.window_size)
    elif args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g,
                                    path_length=args.walk_length,
                                    num_paths=args.number_walks,
                                    dim=args.representation_size,
                                    workers=args.workers,
                                    p=args.p,
                                    q=args.q,
                                    window=args.window_size)
    elif args.method == 'line':
        model = line.LINE(graph=g,
                            epoch=args.epochs,
                            rep_size=args.representation_size,
                            order=args.order,
                            hs=hs)
    
    return model

def merge_embeddings(embeddings1, embeddings2, graph1, graph2, comms, dim=None):
    embeddings = np.zeros((graph1.node_size, dim))

    for comm in comms.keys():
        for node in comms[comm]:
            embeddings[graph1.look_up_dict[node]] = np.concatenate((embeddings1[graph1.look_up_dict[node]], embeddings2[graph2.look_up_dict[comm]]))

    return embeddings

def get_embeddings(graph, embeddings):
    look_back = graph.look_back_list
    vectors = {}
    for i, embedding in enumerate(embeddings):
        vectors[look_back[i]] = embedding

    return vectors

def save_embeddings(args, graph, embeddings, filename):
    vectors = get_embeddings(graph, embeddings)
    fout = open(filename, 'w')
    fout.write('{} {}\n'.format(graph.node_size, np.shape(embeddings)[1]))
    for node, vec in vectors.items():
        fout.write('{} {}\n'.format(str(node), ' '.join([str(x) for x in vec])))

    fout.close()

    return vectors

def main(args):
    g = buildGraph(args)
    print('hierarchical sampling...')
    t1 = time.time()    
    hier_graph = {}
    hier_graph[0] = g
    hier_emb = {}
    comms = {}
    
    k = args.hs_num
    model = buildModel(args, hier_graph[0])
    hier_emb[0] = model.embeddings
    for i in range(k):
        hier_graph[i+1], comms[i] = louvainModularityOptimization(hier_graph[i])       
        model = buildModel(args, hier_graph[i+1], hs=i+1)        
        hier_emb[i+1] = model.embeddings
                
    m = 0
    for j in range(k, 0, -1):      
        hier_emb[j-1] = merge_embeddings(hier_emb[j-1], hier_emb[j], hier_graph[j-1], hier_graph[j], comms[j-1], args.representation_size*(m+2))
        m += 1      
    t2 = time.time()
    print('cost time: %s'%(t2-t1))
    if args.output:
        vectors = get_embeddings(hier_graph[0], hier_emb[0])    
        save_embeddings(args, hier_graph[0], hier_emb[0], args.output)

if __name__ == '__main__':
    random.seed(128)
    np.random.seed(128)
    main(parse_args())
    