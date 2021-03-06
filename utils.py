import numps as np
import torch
import torch.nn as nn
import torch.nn.init as init
import scipy.sparse as sp


def sampling(src_nodes, sample_num, neighbor_tab):
	""" 使用又放回的采样，采样的数量为sample_num，如果邻居数少于采样数，则会出现重复的样本
	"""
	results = []
	for src_id in src_nodes:
		res = np.random.choice(neighbor_tab[src_id], size=(sample_num,))
		results.append(res)
	return np.asarray(results).flatten() # 展开成一维向量


def multihop_sampling(src_nodes, sample_nums, neighbor_tab):
	sampling_result = [src_nodes] # 为什么这样初始化?因为0阶邻居就是它本身
	for k, hopk_num in enumerate(sample_nums):
		hopk_result = sampling(sampling_result[k], hopk_num, neighbor_tab)
		sampling_result.append(hopk_result)

	return sampling_result


# codes below are used for processing data


def encode_onehot(labels):
    classes = set(labels) # 得到所有类别，利用set去重
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)} # 为类别分配one-hot编码
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./datasets/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # genfromtxt会从
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)} # 每篇论文的索引是多少
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), # flatten展开成一维向量
                     dtype=np.int32).reshape(edges_unordered.shape) # 将id相对应的边，改成索引相对应的边。将edges_unordered.flatten()中的值，输入get函数中，返回value
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), # (edges[:, 0], edges[:, 1])这些位置的值为1
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 利用edges生成邻接表
    adj_table = dict()
    for i in range(len(idx)):
        adj_table[i] = []

    for edge in edges:
        if edge[1] not in adj_table[edge[0]]:
            adj_table[edge[0]].append(edge[1])


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features) # 对邻接矩阵采用均值归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)



    return adj, features, labels, idx_train, idx_val, idx_test, adj_table


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
