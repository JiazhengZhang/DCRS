from pathlib import Path
import torch, gc
import time
import logging
import pickle
import logging
import os
import igraph
import networkx as nx
from glob import glob
import  argparse
import random
import numpy as np
import scipy.sparse as sp
from model import *
from yaml import SafeLoader
from utils import *


def CreateGraph(filename):
    G = nx.Graph()
    for line in open(filename):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    return G








def LoadRoleGraph(network, k):
    path_to_train = os.path.join(f".\\data\\realworld\\{network}\\knn\\c{k}" + '.txt')
    struct_edges = np.genfromtxt(path_to_train, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    num_nodes = np.max(sedges) + 1
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(num_nodes, num_nodes),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = Sci_normalize(sadj + sp.eye(sadj.shape[0]))
    Role_Adj = sparse_mx_to_torch_sparse_tensor(nsadj).cuda()

    return Role_Adj






def LoadKnnDGL(network, k):
    path_to_train = os.path.join(f"..\\data\\realworld\\{network}\\knn\\c{k}" + '.txt')
    struct_edges = np.genfromtxt(path_to_train, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    dglknngraph = dgl.graph((sedges[:, 0], sedges[:, 1])).to(torch.device("cuda:0"))
    dglknngraph.add_edges(sedges[:, 1], sedges[:, 0])

    return dglknngraph
















def save_model(net, Network, model_name):
    checkpoint = {"model_state_dict": net.state_dict()}
    path_checkpoint = os.path.join('log', model_name + '_' + FEATURE_TYPE + '_' + OPTIMIOZER,f'{Network}',f'{DATE}', f'debug-{DATE}_{TIME}', f"{model_name}.pt")
    torch.save(checkpoint, path_checkpoint)




def Dismantling(graph,seed_nodes):
    A = graph.get_edgelist()
    G = nx.Graph(A)  # In case you graph is undirected
    original_largest_cc = G.number_of_nodes()
    threshold = 0.01
    TAS_NUM = 0
    TAS_CON = dict.fromkeys(seed_nodes)

    for node in seed_nodes:
        G.remove_node(node)
        if len(G) == 0:
            residual_largest_cc = 0
        else:
            residual_largest_cc = len(max(nx.connected_components(G), key=len)) / original_largest_cc
        TAS_CON[node] = residual_largest_cc

    for node in seed_nodes:
        if float(TAS_CON[node]) > threshold:
            TAS_NUM += 1
        else:
            break




    return TAS_NUM

def Train(Network, log_path, knn, fuse, seed, learning_rate, weight_decay):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)  # reproducibility
        patience = 12
        path_checkpoint = log_path
        early_stopping = EarlyStopping(logging,patience=patience, verbose=True, path=path_checkpoint)


        train_graph, train_dglgraph, train_adj_matrices  = LoadRealworldData(Network, num_feat)
        role_adj = LoadRoleGraph(Network, k=knn)
        model = DCRS ( fuse, num_feat, num_hid).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

        print(f"Network: {Network}, model: {model_name}, Feature: {FEATURE_TYPE}, Knn: {knn}, fuse: {fuse}")

        best_Tperf = 1
        best_epoch = 0



        for epoch in range(500):
            optimizer.zero_grad()
            score = model(train_dglgraph, role_adj, train_dglgraph.ndata['feat']).squeeze(1)
            loss = model.loss(score, train_adj_matrices.to(device), gamma=gamma, mean=False)

            _, TAS_indices = torch.topk(score, score.numel())

            TAS = Dismantling(train_graph, TAS_indices.cpu().numpy())
            TAS_ratio = TAS / score.numel()



            if best_Tperf > TAS_ratio:
                best_Tperf = TAS_ratio
                best_epoch = epoch


            early_stopping(TAS_ratio, model, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(
                f"Train Epoch {epoch} | Loss: {loss:.4f} | TAS: {TAS}/{score.numel()}={TAS_ratio:.6f} ")

            loss.backward()
            optimizer.step()

        return {best_epoch: best_Tperf}






if __name__ == '__main__':



    model_name = 'DCRS'

    List = ['DNCEmails', 'PPI', 'LastFM' ]

    for network in List:
        config = yaml.load(open('config_realworld.yaml'), Loader=SafeLoader)[network]
        feature_type = '1'
        FEATURE_TYPE = "1"
        seed = int(config['seed'])
        knn = config['knn']
        fuse = config['fuse']
        gamma = int(config['gamma'])
        learning_rate = config['learning_rate']
        weight_decay = config['weight_decay']
        patience = 12
        num_feat = config['num_feats']
        num_hid = config['num_hidden']
        title = os.path.join('log', model_name)
        DATE = time.strftime('%m-%d', time.localtime())
        TIME = time.strftime('%H.%M.%S', time.localtime())
        log_path = os.path.join(title, f'{network}', f'{DATE}', f'debug-{DATE}_{TIME}')
        os.makedirs( log_path, exist_ok=True)
        Train(network, log_path, knn, fuse, seed, learning_rate, weight_decay)