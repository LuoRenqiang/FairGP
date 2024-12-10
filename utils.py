import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import networkx as nx
from community import community_louvain
import numpy as np
# import scipy.sparse as sp
import igraph as ig
import leidenalg
import sys
import random
import dgl
import time
import pandas as pd
import argparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj, get_self_loop_attr, scatter, to_edge_index, to_torch_coo_tensor, to_networkx, to_scipy_sparse_matrix
from torch_scatter import scatter_add,scatter_mean
import metis
import heapq
import csv

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j

# random split for ablation
def random_partition(num_nodes, n_patches=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 如果图的节点数小于 n_patches，直接生成一个随机排列的 membership 张量。
    if num_nodes < n_patches:
        membership = torch.randperm(num_nodes)
    else:
        # 随机生成每个节点所属子图的标签。
        membership = torch.randint(0, n_patches, (num_nodes,))

    patch = []
    max_patch_size = -1
    # 对每个子图标签 i，找到所有属于该子图的节点索引，并添加到 patch 列表中。
    for i in range(n_patches): # number of subgraph
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        # 更新 max_patch_size 为最大子图大小
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)): # 填充为最大长度，小于长度的补上虚拟节点
        l = len(patch[i])
        # 如果其大小小于 max_patch_size，用无效节点索引（num_nodes）
        if l < max_patch_size:
            patch[i] += [num_nodes] * (max_patch_size - l) # 虚拟节点

    patch = torch.tensor(patch) # [n_patches, max_patch_size]

    return patch

# metis split
def metis_partition(edge_index, num_nodes, n_patches=50, seed=None):
    # if g['num_nodes'] < n_patches:
    if num_nodes < n_patches:
        # 如果图的节点数小于 n_patches，直接生成一个随机排列的 membership 张量。
        membership = torch.randperm(n_patches)
    else:
        # data augmentation
        # adjlist = g['edge_index'].t()
        adjlist = edge_index.t()
        G = nx.Graph()
        # G.add_nodes_from(np.arange(g['num_nodes']))
        G.add_nodes_from(np.arange(num_nodes))
        G.add_edges_from(adjlist.tolist())
        # metis partition
        # cuts 是分割后的切割数，
        # membership 是每个节点所属子图的标签。
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    # assert len(membership) >= g['num_nodes']
    assert len(membership) >= num_nodes
    # membership = torch.tensor(membership[:g['num_nodes']])
    membership = torch.tensor(membership[:num_nodes])

    patch = []
    max_patch_size = -1
    # 对每个子图标签 i，找到所有属于该子图的节点索引，并添加到 patch 列表中。
    for i in range(n_patches): # number of subgraph
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        # 更新 max_patch_size 为最大子图大小
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)): # 填充为最大长度，小于长度的补上虚拟节点
        l = len(patch[i])
        # 如果其大小小于 max_patch_size，用无效节点索引（g['num_nodes']）
        if l < max_patch_size:
            # patch[i] += [g['num_nodes']] * (max_patch_size - l) # 虚拟节点
            patch[i] += [num_nodes] * (max_patch_size - l) # 虚拟节点

    patch = torch.tensor(patch) # [n_patches, max_patch_size]

    return patch

# louvain split
def louvain_partition(edge_index, num_nodes, n_patches=50, seed=None):
    adjlist = edge_index.t()
    G = nx.Graph()
    G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(adjlist.tolist())

    partition = community_louvain.best_partition(G)
    membership = np.array([partition[i] for i in range(num_nodes)])

    unique_labels, counts = np.unique(membership, return_counts=True)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    membership = np.array([label_map[label] for label in membership])
    print('louvain subgraph num:',len(unique_labels))
    # Split large subgraphs
    max_nodes_per_patch = num_nodes // n_patches
    partition_groups = {label: np.where(membership == label)[0].tolist() for label in unique_labels}
    new_groups = []

    items_to_modify = []

    for group_i, nodes in partition_groups.items():
        while len(nodes) > max_nodes_per_patch:
            long_group = list.copy(nodes)
            partition_groups[group_i] = list.copy(long_group[:max_nodes_per_patch])
            new_grp_i = max(partition_groups.keys()) + 1
            new_groups.append(new_grp_i)
            items_to_modify.append((new_grp_i, long_group[max_nodes_per_patch:]))
            nodes = long_group[max_nodes_per_patch:]

    for new_grp_i, new_nodes in items_to_modify:
        partition_groups[new_grp_i] = new_nodes

    unique_labels = list(partition_groups.keys())
    # print('louvain subgraph 1:',len(unique_labels))
    # Merge communities to get exactly n_patches
    if len(unique_labels) > n_patches:
        print('original communities num: ', len(unique_labels))
        
        # Use a min-heap to efficiently get the smallest community
        community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
        heapq.heapify(community_sizes)
        
        while len(unique_labels) > n_patches:
            smallest_community_size, smallest_community = heapq.heappop(community_sizes)
            smallest_community_members = partition_groups.pop(smallest_community)
            unique_labels.remove(smallest_community)
            
            # Find the closest community with the smallest size
            closest_community = min(
                unique_labels,
                key=lambda x: (
                    len(set(smallest_community_members) & set(partition_groups[x])) + len(partition_groups[x])
                )
            )
            
            partition_groups[closest_community].extend(smallest_community_members)
            unique_labels = list(partition_groups.keys())
            
            # Rebuild the min-heap with updated sizes
            community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
            heapq.heapify(community_sizes)
    else:  # make n_patches equals len(unique_labels)
        print("communities num < n_patch number, modified the n_patch")
        n_patches = len(unique_labels)

    patch = []
    max_patch_size = -1
    for label in unique_labels:
        patch_i = partition_groups[label]
        patch.append(patch_i)
        max_patch_size = max(max_patch_size, len(patch_i))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i].extend([num_nodes] * (max_patch_size - l))

    patch = torch.tensor(patch)
    return patch

# label propagation split 
def label_propagation_partition(edge_index, num_nodes, n_patches=50, seed=None):
    adjlist = edge_index.t()
    G = nx.Graph()
    G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(adjlist.tolist())

    communities = nx.algorithms.community.label_propagation.label_propagation_communities(G)
    membership = [-1] * num_nodes
    for i, community in enumerate(communities):
        for node in community:
            membership[node] = i

    unique_labels = list(set(membership))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    membership = [label_map[label] for label in membership]
    print('label propagation subgraph num:',len(unique_labels))
    # Split large subgraphs
    max_nodes_per_patch = num_nodes // n_patches
    partition_groups = {label: [i for i, x in enumerate(membership) if x == label] for label in unique_labels}
    new_groups = []
    items_to_modify = []

    for group_i, nodes in partition_groups.items():
        while len(nodes) > max_nodes_per_patch:
            long_group = list.copy(nodes)
            partition_groups[group_i] = list.copy(long_group[:max_nodes_per_patch])
            new_grp_i = max(partition_groups.keys()) + 1
            new_groups.append(new_grp_i)
            items_to_modify.append((new_grp_i, long_group[max_nodes_per_patch:]))
            nodes = long_group[max_nodes_per_patch:]

    for new_grp_i, new_nodes in items_to_modify:
        partition_groups[new_grp_i] = new_nodes

    unique_labels = list(partition_groups.keys())

    # Merge communities to get exactly n_patches
    if len(unique_labels) > n_patches:
        print('original communities num: ', len(unique_labels))
        
        # Use a min-heap to efficiently get the smallest community
        community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
        heapq.heapify(community_sizes)
        
        while len(unique_labels) > n_patches:
            smallest_community_size, smallest_community = heapq.heappop(community_sizes)
            smallest_community_members = partition_groups.pop(smallest_community)
            unique_labels.remove(smallest_community)
            
            # Find the closest community with the smallest size
            closest_community = min(
                unique_labels,
                key=lambda x: (
                    len(set(smallest_community_members) & set(partition_groups[x])) + len(partition_groups[x])
                )
            )
            
            partition_groups[closest_community].extend(smallest_community_members)
            unique_labels = list(partition_groups.keys())
            
            # Rebuild the min-heap with updated sizes
            community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
            heapq.heapify(community_sizes)
    else:  # make n_patches equals len(unique_labels)
        print("communities num < n_patch number, modified the n_patch")
        n_patches = len(unique_labels)

    patch = []
    max_patch_size = -1
    for label in unique_labels:
        patch_i = partition_groups[label]
        patch.append(patch_i)
        max_patch_size = max(max_patch_size, len(patch_i))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i].extend([num_nodes] * (max_patch_size - l))

    patch = torch.tensor(patch)
    return patch

# fast label propagation split 
def fast_label_propagation_partition(edge_index, num_nodes, n_patches=50, seed=None):
    adjlist = edge_index.t()
    G = nx.Graph()
    G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(adjlist.tolist())

    communities = nx.algorithms.community.label_propagation.fast_label_propagation_communities(G)
    membership = [-1] * num_nodes
    for i, community in enumerate(communities):
        for node in community:
            membership[node] = i

    unique_labels = list(set(membership))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    membership = [label_map[label] for label in membership]
    print('label propagation subgraph num:',len(unique_labels))
    # Split large subgraphs
    max_nodes_per_patch = num_nodes // n_patches
    partition_groups = {label: [i for i, x in enumerate(membership) if x == label] for label in unique_labels}
    new_groups = []
    items_to_modify = []

    for group_i, nodes in partition_groups.items():
        while len(nodes) > max_nodes_per_patch:
            long_group = list.copy(nodes)
            partition_groups[group_i] = list.copy(long_group[:max_nodes_per_patch])
            new_grp_i = max(partition_groups.keys()) + 1
            new_groups.append(new_grp_i)
            items_to_modify.append((new_grp_i, long_group[max_nodes_per_patch:]))
            nodes = long_group[max_nodes_per_patch:]

    for new_grp_i, new_nodes in items_to_modify:
        partition_groups[new_grp_i] = new_nodes

    unique_labels = list(partition_groups.keys())

    # Merge communities to get exactly n_patches
    if len(unique_labels) > n_patches:
        print('original communities num: ', len(unique_labels))
        
        # Use a min-heap to efficiently get the smallest community
        community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
        heapq.heapify(community_sizes)
        
        while len(unique_labels) > n_patches:
            smallest_community_size, smallest_community = heapq.heappop(community_sizes)
            smallest_community_members = partition_groups.pop(smallest_community)
            unique_labels.remove(smallest_community)
            
            # Find the closest community with the smallest size
            closest_community = min(
                unique_labels,
                key=lambda x: (
                    len(set(smallest_community_members) & set(partition_groups[x])) + len(partition_groups[x])
                )
            )
            
            partition_groups[closest_community].extend(smallest_community_members)
            unique_labels = list(partition_groups.keys())
            
            # Rebuild the min-heap with updated sizes
            community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
            heapq.heapify(community_sizes)
    else:  # make n_patches equals len(unique_labels)
        print("communities num < n_patch number, modified the n_patch")
        n_patches = len(unique_labels)

    patch = []
    max_patch_size = -1
    for label in unique_labels:
        patch_i = partition_groups[label]
        patch.append(patch_i)
        max_patch_size = max(max_patch_size, len(patch_i))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i].extend([num_nodes] * (max_patch_size - l))

    patch = torch.tensor(patch)
    return patch

# leiden split
def leiden_partition(edge_index, num_nodes, n_patches=50, seed=None):
    edges = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]
    G_igraph = ig.Graph(n=num_nodes, edges=edges, directed=False)

    # Perform Leiden partitioning
    partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
    membership = list(partition.membership)

    unique_labels = list(set(membership))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    membership = [label_map[label] for label in membership]
    print('label propagation subgraph num:',len(unique_labels))
    # Split large subgraphs
    max_nodes_per_patch = num_nodes // n_patches
    partition_groups = {label: [i for i, x in enumerate(membership) if x == label] for label in unique_labels}
    new_groups = []
    items_to_modify = []

    for group_i, nodes in partition_groups.items():
        while len(nodes) > max_nodes_per_patch:
            long_group = list.copy(nodes)
            partition_groups[group_i] = list.copy(long_group[:max_nodes_per_patch])
            new_grp_i = max(partition_groups.keys()) + 1
            new_groups.append(new_grp_i)
            items_to_modify.append((new_grp_i, long_group[max_nodes_per_patch:]))
            nodes = long_group[max_nodes_per_patch:]

    for new_grp_i, new_nodes in items_to_modify:
        partition_groups[new_grp_i] = new_nodes

    unique_labels = list(partition_groups.keys())

    # Merge communities to get exactly n_patches
    if len(unique_labels) > n_patches:
        print('original communities num: ', len(unique_labels))
        
        # Use a min-heap to efficiently get the smallest community
        community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
        heapq.heapify(community_sizes)
        
        while len(unique_labels) > n_patches:
            smallest_community_size, smallest_community = heapq.heappop(community_sizes)
            smallest_community_members = partition_groups.pop(smallest_community)
            unique_labels.remove(smallest_community)
            
            # Find the closest community with the smallest size
            closest_community = min(
                unique_labels,
                key=lambda x: (
                    len(set(smallest_community_members) & set(partition_groups[x])) + len(partition_groups[x])
                )
            )
            
            partition_groups[closest_community].extend(smallest_community_members)
            unique_labels = list(partition_groups.keys())
            
            # Rebuild the min-heap with updated sizes
            community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
            heapq.heapify(community_sizes)
    else:  # make n_patches equals len(unique_labels)
        print("communities num < n_patch number, modified the n_patch")
        n_patches = len(unique_labels)

    patch = []
    max_patch_size = -1
    for label in unique_labels:
        patch_i = partition_groups[label]
        patch.append(patch_i)
        max_patch_size = max(max_patch_size, len(patch_i))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i].extend([num_nodes] * (max_patch_size - l))

    patch = torch.tensor(patch)
    return patch


def partition_patch(node_feat, edge_index, labels, n_patches, num_nodes, load_path=None, method='metis', seed=None): # add a node
    if load_path is not None:
        patch = torch.load(load_path)
    else:
        if n_patches == 1:
            patch = torch.tensor(range(num_nodes + 1)).unsqueeze(dim=0)
        else:
            if method == 'metis':
                patch = metis_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
            elif method == 'louvain':
                patch = louvain_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
            elif method == 'lpa':
                patch = label_propagation_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
            elif method == 'fastlpa':
                patch = fast_label_propagation_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
            elif method == 'leiden':
                patch = leiden_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
            elif method == 'random':
                patch = random_partition(num_nodes=num_nodes, n_patches=n_patches)
            else:
                raise ValueError(f"Unknown partitioning method: {method}")
        print('metis done!!!')
    print('patch done!!!')
    # graph['num_nodes'] += 1
    num_nodes += 1
    # graph['node_feat'] = F.pad(graph['node_feat'], [0, 0, 0, 1])
    node_feat = F.pad(node_feat, [0, 0, 0, 1])
    labels = F.pad(labels, [0, 1])
    return patch, node_feat, labels, num_nodes

# combinatorial upper and lower bounds on ORC
def get_local_curvature_profile(edge_index, num_nodes):
    """
    Compute orc approximation. Serves as default, especially for large graphs/ datasets.
    """
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    graph = to_networkx(data)
        
    neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]

    min_orc = []
    max_orc = []

    def compute_upper_bound(node_1, node_2):
        deg_node_1 = len(neighbors[node_1])
        deg_node_2 = len(neighbors[node_2])
        num_triangles = len([neighbor for neighbor in neighbors[node_1] if neighbor in neighbors[node_2]])
        return num_triangles / np.max([deg_node_1, deg_node_2])

    def compute_lower_bound(node_1, node_2):
        deg_node_1 = len(neighbors[node_1])
        deg_node_2 = len(neighbors[node_2])
        num_triangles = len([neighbor for neighbor in neighbors[node_1] if neighbor in neighbors[node_2]])
        return -np.max([0, 1 - 1/deg_node_1 - 1/deg_node_2 - num_triangles/np.min([deg_node_1, deg_node_2])]) - np.max([0, 1 - 1/deg_node_1 - 1/deg_node_2 - num_triangles/np.max([deg_node_1, deg_node_2])]) + num_triangles/np.max([deg_node_1, deg_node_2])

    for node in graph.nodes():
        if len(neighbors[node]) > 0:
            min_orc.append(min([compute_lower_bound(node, neighbor) for neighbor in neighbors[node]]))
            max_orc.append(max([compute_upper_bound(node, neighbor) for neighbor in neighbors[node]]))
        else:
            min_orc.append(0)
            max_orc.append(0)
                                                                    
    lcp_pe = torch.tensor([min_orc, max_orc]).T.float()
    return lcp_pe

# random walk position encoding
def compute_rw_matrix(num_nodes, edge_index, edge_weight):
    # 确保所有边权重为正值
    edge_weights = edge_weight
    assert torch.all(edge_weights > 0)

    # 计算每个节点的加权入度
    edge_indices = edge_index
    rec = edge_indices[0]
    node_deg = scatter(edge_weights, rec, dim_size=num_nodes, reduce='sum')
    pos_idx = torch.where(node_deg > 0)
    node_deg[pos_idx] = 1.0 / node_deg[pos_idx]

    # 将边索引和边权重转换为COO格式的稀疏矩阵
    adj = to_torch_coo_tensor(edge_indices, edge_weights, size=(num_nodes, num_nodes))

    # 归一化邻接矩阵
    adj = adj * node_deg.resize(num_nodes, 1)
    return adj

def generate_random_walk_pe(num_nodes, edge_index, walk_length=10):
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)

    adj = compute_rw_matrix(num_nodes, edge_index, edge_weight)
    out = adj
    pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=num_nodes)]
    
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(get_self_loop_attr(*to_edge_index(out), num_nodes=num_nodes))
        
    pe = torch.stack(pe_list, dim=1)
    return pe

# local degree profile encoding
def compute_local_degree_profile(edge_index, num_nodes):
    # assert data.edge_index is not None
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float).view(-1, 1)
    aggr = FusedAggregation(['min', 'max', 'mean', 'std'])
    xs = [deg] + aggr(deg[col], row, dim_size=num_nodes)

    return xs


def check_nan(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if (value != value).any():  # 检查张量中是否有nan
                print(f"发现nan值在键 {key} 对应的张量中")
                return True
        elif isinstance(value, dict):
            if check_nan(value):  # 递归检查嵌套的字典
                return True
    return False



def jenson_shannon_divergence(net_1_probs, net_2_probs):
    total_m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(net_1_probs.log(), total_m, reduction="batchmean") 
    loss += F.kl_div(net_2_probs.log(), total_m, reduction="batchmean") 
    return (0.5 * loss)


def get_optimizer(optname):
    optimizer = None
    if optname == 'adam':
        optimizer = torch.optim.Adam
    elif optname == 'adamw':
        optimizer = torch.optim.AdamW
    elif optname == 'sgd':
        optimizer = torch.optim.SGD
    elif optname == 'adadelta':
        optimizer = torch.optim.Adadelta
    elif optname == 'adagrad':
        optimizer = torch.optim.Adagrad
    elif optname == 'rmsprop':
        optimizer = torch.optim.RMSprop 
    elif optname == 'radam':
        optimizer = torch.optim.RAdam
        # optimizer = RAdam
    elif optname == 'nadam':
        optimizer = torch.optim.NAdam 
        # optimizer = NAdam
    else:
        raise NotImplementedError
    return optimizer


def validate(model, features, edge_index, labels, sens, mask): # 
    model.eval()
    with torch.no_grad():
        # mask = idx_test if mode == 'test' else idx_val
        logits = model(features, edge_index)
        
        loss = F.cross_entropy(logits[mask], labels[mask]).item()
        acc = accuracy(logits[mask], labels[mask]).item()
        sp, eo = fair_metric(labels, sens, torch.argmax(logits, dim=1), mask)
        
        return loss, acc, sp, eo


# get data size
def get_size_mb(obj):  
    size_bytes = sys.getsizeof(obj)  
    size_mb = size_bytes / (1024 * 1024)  # 1MB = 1024KB = 1024*1024B  
    return size_mb

# def str2bool(v):
#   return v.lower() in ['true', 't']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# 
def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

# 
def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True) # dir name
    fpath = os.path.join(base_dir, filename) # dirname + filename 
    torch.save(data, fpath)


# 待改，增加划分的比例控制
def load_fair_dataset(args):
    datapath = args.datapath
    dataname = args.dataset +'/'
    label_number = args.label_number
    
    print('feat_norm:',args.feat_norm, '\nlabel_number:',label_number,'\nsens_idex:',args.sens_idex,'\nself_loop:',args.self_loop)
    if args.dataset=='nba':
        # edge_df = pd.read_csv('../data/nba/' + 'nba_relationship.txt', sep='\t')
        edges_unordered = np.genfromtxt(datapath + dataname + 'nba_relationship.txt').astype('int')
        # node_df = pd.read_csv(os.path.join('../dataset/nba/', 'nba.csv'))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'nba.csv'))
        print('load edge data')
        predict_attr = 'SALARY'
        labels = idx_features_labels[predict_attr].values
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        sens_attr = "country"
        # labels = y
        adj_start = time.time()
        # feature = node_df[node_df.columns[2:]]
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
            
        column_index = feature.columns.get_loc(sens_attr)  
        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('adj created!')
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] = 1

        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 200
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index # 不包含label [0,1(大于1的转成1)]以外的值的id

    elif args.dataset=='pokec_z': # sens (gender, age, education, region)
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        # sens_attr = 'region' # gender AGE
        sens_attr = args.sens_attr
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])

        column_index = feature.columns.get_loc(sens_attr)  
        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        if sens_attr == 'AGE':
            sens = (sens>=25)
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 50000 # for fedgnn full
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='pokec_n':
        edges_unordered = np.genfromtxt(datapath + dataname + 'region_job_2_relationship.txt').astype('int')
        predict_attr = 'I_am_working_in_field'
        # sens_attr = 'region'
        sens_attr = args.sens_attr
        print('Loading {} dataset'.format(args.dataset))
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'region_job_2.csv'))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)
        feature = idx_features_labels[header]
        # feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        #-----
        adj_start = time.time()
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #将数据拆分成edges_unordered大小的行数的矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        if sens_attr == 'AGE':
            sens = (sens>=25)
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        # return feature
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 1000
        # label_number = 50000 # for fedgnn full
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,args.seed)
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='credit':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'credit.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'credit_edges.txt').astype('int')
        sens_attr="Age"
        predict_attr="NoDefaultNextMonth"
        print('Loading {} dataset'.format(args.dataset))
        # header = list(idx_features_labels.columns)
        header = list(idx_features_labels.columns)
        header.remove('Single')
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        # label_number = 6000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 6000')
            label_number = 6000
        # label_number = 20000 # for fedgnn full
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
       
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='income':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'income.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'income_edges.txt').astype('int')
        sens_attr="race"
        predict_attr="income"
        print('Loading {} dataset'.format(args.dataset))
        header = list(idx_features_labels.columns) #list将括号里的内容变为数组
        header.remove(predict_attr) #header.remove删除括号内的东西
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
            
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) 
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        # feature = feature_normalize(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        labels[labels >1] =1
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        if label_number>=feature.shape[0]:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='german':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'german.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'german_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="Gender"
        predict_attr="GoodCustomer"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        feature = idx_features_labels[header]
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)  
        
        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        feature = sp.csr_matrix(feature, dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature.todense())
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 100')
            label_number = 100
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='bail':
        idx_features_labels = pd.read_csv(os.path.join(datapath + dataname, 'bail.csv'))
        edges_unordered = np.genfromtxt(datapath + dataname + 'bail_edges.txt').astype('int')
        print('Loading {} dataset'.format(args.dataset))
        sens_attr="WHITE"
        predict_attr="RECID"
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        
        feature = idx_features_labels[header]
        labels = idx_features_labels[predict_attr].values #存下predict_attr的数值
        
        if args.sens_idex:
            print('remove sensitive from node attribute')
            feature = feature.drop(columns = [sens_attr])
        column_index = feature.columns.get_loc(sens_attr)
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        sens = idx_features_labels[sens_attr].values.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feature)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        # feature = feature_normalize_column(feature)
        feature = torch.FloatTensor(feature)
        labels = torch.LongTensor(labels) 
        
        print('dataset:',args.dataset)
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        # label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index

    elif args.dataset=='facebook':
        # datapath + dataname
        sens_attr="gender"
        # predict_attr="RECID"
        edges_file = open(os.path.join(datapath, dataname, "107.edges"))
        edges = []
        for line in edges_file:
            edges.append([int(one) for one in line.strip("\n").split(" ")])

        feat_file = open(os.path.join(datapath, dataname, "107.feat"))
        feats = []
        for line in feat_file:
            feats.append([int(one) for one in line.strip("\n").split(" ")])

        feat_name_file = open(os.path.join(datapath, dataname, "107.featnames"))
        feat_name = []
        for line in feat_name_file:
            feat_name.append(line.strip("\n").split(" "))
            
        names = {}
        for name in feat_name:
            if name[1] not in names:
                names[name[1]] = name[1]

        feats = np.array(feats)

        # node_mapping = {}
        idx_map = {}
        for j in range(feats.shape[0]): # 
            idx_map[feats[j][0]] = j

        feats = feats[:, 1:]

        sens = feats[:, 264] # gender
        labels = feats[:, 220]
        # column_index = feature.columns.get_loc(sens_attr)
        feats = np.concatenate([feats[:, :264], feats[:, 266:]], -1) # 0-263||266-
        feats = np.concatenate([feats[:, :220], feats[:, 221:]], -1) # 0-263||266- []

        edges = np.array(edges)
        # node_num = feats.shape[0]

        idx = np.arange(feats.shape[0])
        # idx_map = {feats[j][0]: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges.flatten())),dtype=int).reshape(edges.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
        print('original edge num',adj.sum())
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #相似矩阵
        if args.self_loop:
            print('add self-loop')
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')
        adj_end = time.time()

        sens = sens.astype(int) 
        sens = torch.FloatTensor(sens) 
        feature = np.array(feats)
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        # feature = feature_normalize(feature)
        feature = torch.FloatTensor(feature)

        if args.sens_idex:
            print('remove sensitive from node attribute')
            # feature = feature.drop(columns = ["WHITE"])
            # feature = feature.drop(columns = [sens_attr])
        else:
            feature = torch.cat([feature, sens.unsqueeze(-1)], -1)
        column_index = feature.shape[1]-1
        labels = torch.LongTensor(labels) 

        print('dataset:',dataname)
        print('edges:',adj.sum())
        print('sens:',sens_attr)
        print('feature:',feature.shape)
        print('labels:',torch.unique(labels, return_counts=True))
        # idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20)
        # label_number = 1000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        # return adj, feature, labels, sens, idx_train, idx_val, idx_test
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index

    elif args.dataset=='aminer_s': # multi sens and multi label(label have binarilized)
        sens_attr = 'continent'
        name = 'Small'
        edgelist = csv.reader(open(os.path.join(datapath, dataname, "edgelist_{}.txt".format(name))))

        edges = []
        for line in edgelist:
            edge = line[0].split("\t")
            edges.append([int(one) for one in edge])

        edges = np.array(edges)

        labels_file = csv.reader(open(os.path.join(datapath, dataname, "labels_{}.txt".format(name))))
        labels = []
        for line in labels_file:
            labels.append(float(line[0].split("\t")[1]))
        labels = np.array(labels)

        sens_file = csv.reader(open(os.path.join(datapath, dataname, "sens_{}.txt".format(name))))
        sens = []
        for line in sens_file:
            sens.append([float(line[0].split("\t")[1])])
        sens = np.array(sens)
        sens = torch.FloatTensor(sens)
        
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        # adj = adj + sp.eye(adj.shape[0])
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')

        feature = np.load(os.path.join(datapath, dataname, "X_{}.npz".format(name)))
        feature = sp.coo_matrix(
                        (feature["data"], (feature["row"], feature["col"])),
                        shape=(labels.shape[0], np.max(feature["col"]) + 1),
                        dtype=np.float32,
                    ).todense()
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        
        feature = torch.FloatTensor(feature)
        feature = torch.cat([feature, sens], -1)
        sens = sens.squeeze()
        column_index = feature.shape[1]-1
        node_num = labels.shape[0]

        unique_labels, counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        labels = (labels == most_common_label).astype(int)
        
        labels = torch.LongTensor(labels) 
        # label_number=1000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
    
    elif args.dataset=='aminer_l': # multi sens and multi label(label have binarilized)
        sens_attr = 'continent'
        name = 'LCC'
        edgelist = csv.reader(open(os.path.join(datapath, dataname, "edgelist_{}.txt".format(name))))

        edges = []
        for line in edgelist:
            edge = line[0].split("\t")
            edges.append([int(one) for one in edge])

        edges = np.array(edges)

        labels_file = csv.reader(open(os.path.join(datapath, dataname, "labels_{}.txt".format(name))))
        labels = []
        for line in labels_file:
            labels.append(float(line[0].split("\t")[1]))
        labels = np.array(labels)

        sens_file = csv.reader(open(os.path.join(datapath, dataname, "sens_{}.txt".format(name))))
        sens = []
        for line in sens_file:
            sens.append([float(line[0].split("\t")[1])])
        sens = np.array(sens)
        sens = torch.FloatTensor(sens)
        
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # feature = normalize(feature)
        # adj = adj + sp.eye(adj.shape[0])
        if args.self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye对角线上位1的矩阵
        else:
            print('no add self-loop')

        feature = np.load(os.path.join(datapath, dataname, "X_{}.npz".format(name)))
        feature = sp.coo_matrix(
                        (feature["data"], (feature["row"], feature["col"])),
                        shape=(labels.shape[0], np.max(feature["col"]) + 1),
                        dtype=np.float32,
                    ).todense()
        if args.feat_norm=='row':
            feature = feature_normalize(feature)
        elif args.feat_norm=='column':
            feature = feature_normalize_column(feature)
        elif args.feat_norm=='none':
            pass
        
        feature = torch.FloatTensor(feature)
        feature = torch.cat([feature, sens], -1)
        sens = sens.squeeze()
        column_index = feature.shape[1]-1
        node_num = labels.shape[0]

        unique_labels, counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        labels = (labels == most_common_label).astype(int)
        
        labels = torch.LongTensor(labels) 
        # label_number=1000
        if label_number>=feature.shape[0]*0.5:
            print('label_numberis larger than node number! set to 1000')
            label_number = 1000
        idx_train, idx_val, idx_test = train_val_test_split(labels,0.5,0.25,20,label_number)
        
        return adj, feature, labels, sens, idx_train, idx_val, idx_test, column_index
 


# 只对可用的node id进行选取切分
def train_val_test_split(labels,train_ratio=0.5,val_ratio=0.25,seed=20,label_number=1000):
    import random
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]  # 只要label为0和1的
    label_idx_1 = np.where(labels==1)[0]  # 
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)
    position1 = train_ratio
    position2 = train_ratio + val_ratio
    idx_train = np.append(label_idx_0[:min(int(position1 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(position1 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(position1 * len(label_idx_0)):int(position2 * len(label_idx_0))], 
                        label_idx_1[int(position1 * len(label_idx_1)):int(position2 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(position2 * len(label_idx_0)):],
                         label_idx_1[int(position2 * len(label_idx_1)):])
    print('train,val,test:',len(idx_train),len(idx_val),len(idx_test))
    return idx_train, idx_val, idx_test

# sp mx 2 edge_index
def sparse_2_edge_index(adj):   
    edge_index_origin = adj.nonzero()
    edge_index = torch.stack([torch.from_numpy(edge_index_origin[0]).long(), torch.from_numpy(edge_index_origin[1]).long()])
    return edge_index 

# dense mx 2 edge_index
def dense_2_edge_index(adj):
    edge_index = adj.nonzero().t().contiguous()
    return edge_index

# 防止数值过小
def non_small_zero(number):
    number = number+1e-6 if number==0 else number
    return number

# mask [True, Fasle, False]
def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx]=True
    return mask

# idx [0]
def mask2idx(mask):
    return torch.nonzero(mask, as_tuple=False).squeeze()

# 
def fair_metric(y, sens, output, idx):
    if idx.dtype == torch.bool:
        idx = torch.nonzero(idx, as_tuple=False).squeeze()
    val_y = y[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)
    
    pred_y = None
    if len(idx)==1: # only one sample
        pred_y = (output[idx] > 0).type_as(y).cpu().numpy() # array([1, 0])
    else:
        pred_y = (output[idx].squeeze() > 0).type_as(y).cpu().numpy() # 1 or 0
    
    sum_idx_s0, sum_idx_s1 = non_small_zero(sum(idx_s0)), non_small_zero(sum(idx_s1))
    sum_idx_s0_y1, sum_idx_s1_y1 = non_small_zero(sum(idx_s0_y1)), non_small_zero(sum(idx_s1_y1))

    parity = abs(sum(pred_y[idx_s0]) /sum_idx_s0 - sum(pred_y[idx_s1]) / sum_idx_s1)
    equality = abs(sum(pred_y[idx_s0_y1]) / sum_idx_s0_y1 - sum(pred_y[idx_s1_y1]) / sum_idx_s1_y1)

    return parity, equality


# 
def fair_metric_new(y, sens, output, idx):
    if idx.dtype == torch.bool:
        idx = torch.nonzero(idx, as_tuple=False).squeeze()
    val_y = y[idx].cpu().numpy()
    val_sens = sens[idx].cpu().numpy()

    unique_sens = np.unique(val_sens)
    
    # 预测值
    if len(idx) == 1:  # 仅一个样本
        pred_y = (output[idx] > 0).type_as(y).cpu().numpy()  # array([1, 0])
    else:
        pred_y = (output[idx].squeeze() > 0).type_as(y).cpu().numpy()  # 1 或 0

    sum_pred_y = []
    sum_true_y = []
    sum_sens = []

    for s in unique_sens:  # 对于每个敏感属性
        idx_s = val_sens == s
        sum_sens.append(non_small_zero(np.sum(idx_s)))
        sum_pred_y.append(np.sum(pred_y[idx_s]) / non_small_zero(np.sum(idx_s)))
        
        # 计算真实标签为1的样本比例
        idx_s_y1 = np.bitwise_and(idx_s, val_y == 1) 
        sum_true_y.append(np.sum(pred_y[idx_s_y1]) / non_small_zero(np.sum(idx_s_y1)))

    # 计算方差
    # parity = np.var(sum_pred_y)
    # equality = np.var(sum_true_y)
    parity = np.std(sum_pred_y)
    equality = np.std(sum_true_y)
    
    return parity, equality

def feature_normalize(feature):  # 行归一化
    '''sum_norm'''
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True) # [num_node]
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum


def feature_normalize_column(feature, feature_range=(0, 1)):  # 列归一化(求和)
    '''sum_norm'''
    feature = np.array(feature)
    data_min = feature.min(axis=0,keepdims=True)  
    data_max = feature.max(axis=0,keepdims=True)  
    rowsum = data_max - data_min  
    # rowsum = feature.sum(axis=1, keepdims=True) # [num_node]
    rowsum = np.clip(rowsum, 1, 1e10)
    scale = (feature_range[1] - feature_range[0]) / rowsum 
    min_max_data = data_min + scale * (feature - data_min) 
    min_max_data = min_max_data * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return min_max_data  


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

# one-hot output , num label
def accuracy(output, labels): # logits,label()
    preds=None
    if len(output.shape)>1:
        preds = output.max(1)[1].type_as(labels)
    else:
        preds = (output > 0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

# edge_index 2 sp matrix
def edge_index_2_sparse_mx(edge_index):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    num_nodes = max(np.max(row), np.max(col)) + 1
    # 使用 scipy.sparse.coo_matrix 构造函数创建 COO 稀疏矩阵
    sparse_adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return  sparse_adj

def edge_index_2_sparse_tensor(edge_index, num_nodes):
    # 构建稀疏矩阵
    values = torch.ones(edge_index.size(1))  # 假设所有边的权重都是1
    sparse_adj = torch.sparse.FloatTensor(edge_index, values, torch.Size([num_nodes, num_nodes]))
    return sparse_adj

# sparse normalize D^{-0.5}(A+I)D^{-0.5}
def sys_normalized_adjacency(adj): # add self_loop
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

# 
def laplace_decomp(g, max_freqs):

    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray()) # 前m小
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        # g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
        EigVecs = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        # g.ndata['EigVecs']= EigVecs
        EigVecs = EigVecs
        
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    #Save EigVals node features
    # g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    EigVals = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    return EigVecs, EigVals

def laplacian_positional_encoding(g, pos_enc_dim):
    # 
    laplacian = sp.csgraph.laplacian(g, normed=False)
    # laplacian = sp.csgraph.laplacian(g, normed=True)
    eignvalue, eignvector = sp.linalg.eigsh(laplacian, which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def adjacency_positional_encoding(g, pos_enc_dim): # coo
    # adj = g.adjacency_matrix_scipy(return_edge_ids=False)
    # adj = g.adj_sparse('coo',return_edge_ids=False)
    # adj = g.adjacency_matrix(scipy_fmt="coo")
    eignvalue, eignvector = sp.linalg.eigsh(g, which='LM', k=pos_enc_dim)
    # eignvalue, eignvector = sp.linalg.eigsh(g.adjacency_matrix_scipy(return_edge_ids=False).astype(float), which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix

def set_seed(seed = 20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_subgraph_adj_by_nodes(adj, nodes): 
    graph = dgl.from_scipy(adj) 
    subgraph = graph.subgraph(nodes) 
    return subgraph.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
