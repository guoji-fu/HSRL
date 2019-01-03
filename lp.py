from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from units import load_embeddings, read_node_label
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def link_pred(filename, embeddings):
    Y_pred = []
    edge_exist = []
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            start_entity, end_entity, edge_status = line.strip().split()      
            if start_entity in embeddings.keys() and end_entity in embeddings.keys():   
                Y_pred.append((cosine_similarity([embeddings[start_entity], embeddings[end_entity]])[0][1] + 1.0) / 2.0)

                edge_exist.append(int(edge_status))
            
            count += 1
    
    return edge_exist, Y_pred

def save_test(filename, y_test, y_score):
    f = open(filename, 'w')
    for i in range(len(y_test)):
        f.write('{}\t{}\n'.format(y_test[i], y_score[i]))
    
    f.close()

def cal_auc(y_test, y_score):
    """
    calculate AUC value and plot the ROC curve
    """
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
    # plt.plot(fpr, tpr, color='black', lw = 1)
    # plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
    # plt.text(0.5,0.3,'ROC curve (area = %0.3f)' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    return roc_auc

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input',
                        default='output/dblp/dblp_hs_dw_1.txt',
                        help=' ')
    args = parser.parse_args()

    return args    

def main():
    args = parse_args()   

    ## movielens 
    embeddings = load_embeddings('output/movielens/movielens_dw_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/movielens/movielens_n2v_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/movielens/movielens_line_hs_lp_embeddings.txt') 

    test_file = 'data/movielens/test_edges.txt' 

    ## DBLP 
    # embeddings = load_embeddings('output/dblp/dblp_n2v_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/dblp/dblp_line_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/dblp/20/dblp_hs_line_lp_20_embeddings.txt')     
    
    # test_file = 'data/dblp/test_edges.txt'
    
    ## Douban 
    # embeddings = load_embeddings('output/douban/douban_dw_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/douban/douban_n2v_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/douban/douban_line_hs_lp_embeddings.txt') 

    # test_file = 'data/douban/test_edges.txt'

    ## Yelp  
    # embeddings = load_embeddings('output/yelp/yelp_dw_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/yelp/yelp_n2v_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/yelp/yelp_line_hs_lp_embeddings.txt') 

    # test_file = 'data/yelp/test_edges.txt'

    ## MIT
    # embeddings = load_embeddings('output/mit/mit_dw_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/mit/mit_n2v_hs_lp_embeddings.txt') 
    # embeddings = load_embeddings('output/mit/mit_line_hs_lp_embeddings.txt') 

    # test_file = 'data/mit/test_edges.txt'

    Y_train, Y_train_pred = link_pred(test_file, embeddings)

    # save_test('lp_results.txt', Y_train, Y_train_pred)  
    print(cal_auc(Y_train, Y_train_pred))
    
if __name__ == '__main__':
    main()