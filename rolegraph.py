import networkx as nx
import numpy as np
import os
from utils import pickle_read, pickle_save
import argparse

from sklearn.metrics.pairwise import cosine_similarity as cos
import multiprocessing as mp


def CreateGraph(filename):
    G = nx.Graph()
    for line in open(filename):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    return G



def construct_synthetic_graph(data_path, dataset, features, topk):

    os.makedirs(os.path.join( 'data', 'synthetic', data_path, f'{dataset}', 'knn'), exist_ok=True)
    fname = f'./data/synthetic/{data_path}/' + dataset + '/knn/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()





def role_graph(data_path,dataset):
    for topk in range(2, 10):
        data = pickle_read(f'./data/synthetic/{data_path}/RoIX_' + dataset + '.npy')
        print(data)
        construct_synthetic_graph(data_path, dataset, data, topk)
        f1 = open(f'./data/synthetic/{data_path}/' + dataset + '/knn/tmp.txt','r')
        f2 = open(f'./data/synthetic/{data_path}/' + dataset + '/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            # if int(start) < int(end):
            f2.write('{} {}\n'.format(start, end))
        f2.close()


def func(z):

    return role_graph(z[1], z[0])


def parse_args():

    parser = argparse.ArgumentParser( description = "Please customize these parameters" )
    parser.add_argument("--Type", '-t', type=str)
    parser.add_argument("--M", '-m', type= str, help="BA_parameter")
    parser.add_argument("--P", '-p', type= str, help="edge_probability")
    parser.add_argument("--N", '-n', type= int, help="Size", default= 1000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    SYN = args.Type
    num_graph = 20
    para_size = args.N
    para_1 = args.M
    para_2 = args.P
    if SYN == 'BA':
        SYN_PARA = f'{SYN}_{para_size}_{para_1}'
    elif SYN == 'ER':
        SYN_PARA = f'{SYN}_{para_size}_{para_2}'
    else:
        SYN_PARA = f'{SYN}_{para_size}_{para_1}_{para_2}'


    List = [str(i) for i in range(0, num_graph)]
    MN = [( i, SYN_PARA) for i in List]

    for i in MN:
        func(i)
    # pool = mp.Pool(processes = 5)
    # res = pool.map(func, MN)
