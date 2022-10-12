import sys
import torch, gc
from model import *
import time
import logging
import pickle
import logging
from utils import EarlyStopping
import os
import igraph
import yaml
from yaml import SafeLoader
from utils import *
import networkx as nx
from glob import glob
import  argparse
from model import *
import random
import numpy as np
import scipy.sparse as sp
import dgl




def CreateGraph(filename):
    G = nx.Graph()
    for line in open(filename):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    return G

def dismantling( network, seed_nodes):
    # A = graph.get_edgelist()
    G = CreateGraph(os.path.join('.', 'data', 'synthetic', SYN_PARA, str(network)+'.edge'))
    # In case you graph is undirected
    original_largest_cc = G.number_of_nodes()


    TAS = []
    TAS_CON = []
    residual_largest_cc = 1
    TAS_NUM = 0
    Threshold = 0.01

    for node in seed_nodes:
        if residual_largest_cc > Threshold and residual_largest_cc != 0 and G.number_of_edges():
            # print(node)
            G.remove_node(node)
            TAS_NUM = TAS_NUM +1
            if len(G) == 0:
                residual_largest_cc = 0
            else:
                residual_largest_cc = len(max(nx.connected_components(G), key=len)) / original_largest_cc
            TAS_CON.append(residual_largest_cc)
            TAS.append(node)


    return TAS_NUM





def Train(id, SYN_PARA, log_save_path):

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = EarlyStopping(logging, patience=patience, verbose=True, path=log_save_path)

    dgl_g, ig_g, feature, g_role_adj, nx_g = LoadSyntheticData(id, SYN_PARA, num_feat, knn, str(feature_type))
    g_adj = dgl_g.adj().to_dense()

    model = DCRS( fuse, num_feat, num_hid).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= weight_decay)


    print(f"Parameters: modelwork: {id}, seed: {seed}, learning rate: {learning_rate:.4}, Feature: {feature_type}, NUM_FEATS: {num_feat}, NUM_HIDDEN: {num_hid}")

    best_Tperf = 1
    best_epoch = 0

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()

        score = model(dgl_g.to(device), g_role_adj.to(device), feature.to(device))
        loss = model.loss(score.squeeze(1), g_adj.to(device), gamma=gamma, mean=lossmean)

        _, TAS_indices = torch.topk(score.squeeze(1), score.numel())
        TAS = dismantling(id, TAS_indices.cpu().numpy())
        TAS_ratio =  TAS / score.numel()

        if best_Tperf > TAS_ratio:
            best_Tperf = TAS_ratio
            best_TAS = TAS
            best_epoch = epoch


        early_stopping(TAS_ratio, model, epoch)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(
            f"Train Epoch {epoch} | Loss: {loss:.4f} | TAS: {TAS}/{score.numel()}={TAS_ratio:.6f} ")

        loss.backward()
        optimizer.step()

    return {best_epoch: best_TAS}






def parse_args():

    parser = argparse.ArgumentParser( description = "Please customize these parameters" )
    parser.add_argument("--Type", '-t', type=str)
    parser.add_argument("--M", '-m', type= str, help="BA_parameter")
    parser.add_argument("--P", '-p', type= str, help="edge_probability")
    parser.add_argument("--N", '-n', type= int, help="Size", default= 1000)
    parser.add_argument('--config', type=str, default='config_synthetic.yaml')

    args = parser.parse_args()
    return args



if __name__ == '__main__':


    args = parse_args()
    SYN = args.Type
    para_size = args.N
    para_1 = args.M
    para_2 = args.P


    if SYN == 'BA':
        SYN_PARA = f'{SYN}_{para_size}_{para_1}'
    elif SYN == 'ER':
        SYN_PARA = f'{SYN}_{para_size}_{para_2}'
    else:
        SYN_PARA = f'{SYN}_{para_size}_{para_1}_{para_2}'



    config = yaml.load(open(args.config), Loader=SafeLoader)[SYN_PARA]

    num_instances = 20
    seed = config['seed']
    knn = config['knn']
    fuse = config['fuse']
    gamma = config['gamma']
    lossmean = config['Lossmean']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    feature_type = str(config['feature_type'])
    patience = 10
    optimizer = config['optimizer']
    num_feat = config['num_feats']
    num_hid = config['num_hidden']
    model_name = config['model']
    title = model_name

    for id in range(num_instances):
        DATE = time.strftime('%m-%d', time.localtime())
        TIME = time.strftime('%H.%M.%S', time.localtime())

        os.makedirs(os.path.join('log', title, SYN_PARA, f'{id}'), exist_ok=True)
        log_save_path = os.path.join('log', title, SYN_PARA, f'{id}')
        Train(str(id), SYN_PARA, log_save_path)