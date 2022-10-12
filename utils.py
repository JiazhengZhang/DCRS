import networkx as nx
import numpy as np
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import igraph
from sklearn.preprocessing import normalize
import pickle
import dgl
import yaml
import numpy as np
import torch
import os

def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logging = logger
    def __call__(self, val_loss, model,epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, f"checkpoint_{epoch}.pt"))
        self.val_loss_min = val_loss

def LoadSyntheticGraph( network, path):
    path_to_train = os.path.join(path, str(network) + '.edge')
    struct_edges = np.genfromtxt(path_to_train, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    dglgraph = dgl.graph((sedges[:, 0], sedges[:, 1]))
    dglgraph.add_edges(sedges[:, 1], sedges[:, 0])

    return dglgraph

def LoadRealWorldGraph( network, path):
    path_to_train = os.path.join(path, str(network) + '.txt')
    struct_edges = np.genfromtxt(path_to_train, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    dglgraph = dgl.graph((sedges[:, 0], sedges[:, 1]))
    dglgraph.add_edges(sedges[:, 1], sedges[:, 0])

    return dglgraph


def LoadRoleAdj(network, path, k, num_nodes):

    path_to_train = os.path.join(path, network, 'knn', f'c{k}.txt')
    struct_edges = np.genfromtxt(path_to_train, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(num_nodes, num_nodes),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = Sci_normalize(sadj + sp.eye(sadj.shape[0]))
    Role_Adj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return Role_Adj


def CreateGraph(network, path):
    G = nx.Graph()
    path = os.path.join(path, str(network) + '.edge')
    for line in open(path):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Sci_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # print(rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    # print(r_inv)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # print(r_mat_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def save_dict_to_yaml(dict_value: dict, save_path: str):
    with open(save_path, 'w', encoding="utf-8") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

def LoadOnefeature(num_nodes, num_feats):

    return torch.ones(num_nodes, num_feats)

def LoadSyntheticData( id, SYN_PARA, num_feats, knn, feature_type='1'):
    datadir = os.path.join('.', 'data', 'synthetic', SYN_PARA)

    if not os.path.exists(datadir):
        os.makedirs(datadir)


    else:
        dgl_g = LoadSyntheticGraph( id, datadir)
        g_role_adj = LoadRoleAdj(id, datadir, knn, dgl_g.num_nodes())
        if feature_type == '1':
            feature = LoadOnefeature(dgl_g.num_nodes(), num_feats)

        nx_g = CreateGraph(id, datadir)
        ig_g = igraph.Graph().Read_Edgelist(os.path.join(datadir, f'{id}.edge'), directed=False)

        return dgl_g, ig_g, feature, g_role_adj, nx_g



def LoadRealworldData( network, num_feats, feature_type='1'):
    datadir = os.path.join('.', 'data', 'realworld', f'{network}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ig_g = igraph.Graph().Read_Edgelist(os.path.join(datadir, f'{network}.txt'), directed=False)
    dgl_g = LoadRealWorldGraph(network, datadir).to(device)
    ig_adj = np.array(ig_g.get_adjacency().data, dtype=bool)
    ig_adj = torch.from_numpy(np.array(ig_adj, dtype=int)).float().cuda()
    ig_adj += torch.eye(ig_g.vcount()).cuda()
    if feature_type == '1':
        dgl_g.ndata['feat'] = LoadOnefeature(dgl_g.num_nodes(), num_feats).to(device)

    return ig_g, dgl_g, ig_adj








#
# def LoadGraph(input_dim, TRAIN_Graph, directed_train=False, feature_type='1', use_cuda= True):
#     path_to_train = os.path.join( f"..\\data\\realworld\\{TRAIN_Graph}"+'.txt')
#     g = igraph.Graph().Read_Edgelist(
#         path_to_train, directed=directed_train)
#     dg, feature_dim = get_rev_dgl( network=TRAIN_Graph, graph=g, feature_type=feature_type, feature_dim=input_dim, use_cuda=use_cuda)
#
#
#
#     return g, dg, feature_dim

