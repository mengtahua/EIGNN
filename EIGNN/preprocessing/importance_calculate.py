import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import random

def estimate_diameter_sample(G, samples=5, max_cap=50):

    n = G.number_of_nodes()
    if n == 0:
        return 0
    samples = min(samples, n)
    nodes = list(G.nodes())
    max_dist = 1
    for v in random.sample(nodes, samples):
        lengths = nx.single_source_shortest_path_length(G, v)
        if lengths:
            cur_max = max(lengths.values())
            if cur_max > max_dist:
                max_dist = cur_max
        if max_dist >= max_cap:
            break
    return min(max_dist, max_cap)

def phi_generate_sparse(A_in, time_steps=12, mix=0.1, support_matrix=None, eps=1e-12):
    """
    A_in: scipy.sparse 
    support_matrix:
    """
    if not sp.isspmatrix_csr(A_in):
        A_in = csr_matrix(A_in)
    N = A_in.shape[0]

    # 拓扑支撑：binary csr
    if support_matrix is None:
        A_sup = (A_in != 0).astype(np.float64).tocsr()
    else:
        A_sup = csr_matrix((support_matrix != 0).astype(np.float64))


    deg = np.asarray(A_sup.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0


    max_allowed = min(time_steps, 50)
    ts = max_allowed

    node_feat = np.zeros((N, 2), dtype=np.float64)

    for step in range(ts + 2):
        deg_next = A_sup.dot(deg)    
        if step >= ts:
            node_feat[:, step - ts] = deg_next
        deg = deg_next


    A_sup_coo = A_sup.tocoo()
    rows = A_sup_coo.row
    cols = A_sup_coo.col

    num = node_feat[cols, :]          # (E, 2)
    den = node_feat[rows, :] + eps    # (E, 2)
    w = (num / den).sum(axis=1)       # (E,)

    W = csr_matrix((w, (rows, cols)), shape=(N, N))

    
    M = W + (A_in.multiply(mix))

    #sparse
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv_rs = 1.0 / row_sums
    D_inv = sp.diags(inv_rs)
    A_new = D_inv.dot(M)
    return A_new.tocsr()

def shano_sparse(A_new_csr, A_orig_csr, eps=1e-12):
    """
    sparse shano：return dict(idx->score)
    """
    if not sp.isspmatrix_csr(A_new_csr):
        A_new_csr = csr_matrix(A_new_csr)
    if not sp.isspmatrix_csr(A_orig_csr):
        A_orig_csr = csr_matrix(A_orig_csr)

    N = A_new_csr.shape[0]

    A_mask = A_orig_csr.tocoo()
    rows = A_mask.row
    cols = A_mask.col

    
    A_new_coo = A_new_csr.tocoo()
    idx_map = {(int(r), int(c)): float(v) for r, c, v in zip(A_new_coo.row, A_new_coo.col, A_new_coo.data)}

    A1_contrib_row = np.zeros(N, dtype=np.float64)
    A2_contrib_row = np.zeros(N, dtype=np.float64)

    def safe_log2(x):
        return np.log2(x + eps)

    for r, c in zip(rows, cols):
        a1 = idx_map.get((r, c), 0.0)   
        if a1 > 0:
            A1_contrib_row[r] += -(a1 * safe_log2(a1))
        a2 = idx_map.get((c, r), 0.0)   
        if a2 > 0:
            A2_contrib_row[r] += -(a2 * safe_log2(a2))

    A_total = A1_contrib_row + A2_contrib_row
    return {i: float(A_total[i]) for i in range(N)}

def lamda2_caculate(G, edge_list, lenth=None):
    """
     lamda2_caculate：return edge_importance, node_importance
    """
    #if use_estimate_diameter:
        #try:
            #lenth = estimate_diameter_sample(G, samples=diameter_samples, max_cap=50)
        #except Exception:
            #lenth = 12
    #else:
        #try:
            #lenth = nx.diameter(G)
        #except Exception:
            #lenth = estimate_diameter_sample(G, samples=diameter_samples, max_cap=50)
    if lenth==None:
        G_sub = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        lenth = nx.diameter(G_sub)
    print('lenth：', lenth)

    nodes = list(G.nodes())
    node_id_map = {node: idx for idx, node in enumerate(nodes)}
    inv_map = {idx: node for node, idx in node_id_map.items()}
    N = len(nodes)


    A_sparse = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=np.float64)


    out_degree = np.asarray(A_sparse.sum(axis=1)).ravel()
    out_degree[out_degree == 0] = 1.0

    # A_norm = (A / out_degree[:, None]).T 
    inv_out = 1.0 / out_degree
    D_inv = sp.diags(inv_out)
    A_norm = (D_inv.dot(A_sparse)).transpose().tocsr()

  
    A_new = phi_generate_sparse(A_norm, time_steps=lenth, mix=0.1, support_matrix=A_sparse)

    # 加权 A_weighted = out_degree[:, None] * A_new
    D_out = sp.diags(out_degree)
    A_weighted = D_out.dot(A_new).tocsr()
    D_out = csr_matrix(D_out)

    edge_importance = {}
    node_importance = {}
    for src, dst in edge_list:
        if src not in node_id_map or dst not in node_id_map:
            continue
        s_idx = node_id_map[src]
        d_idx = node_id_map[dst]

        try:
            val_sd = float(A_weighted[s_idx, d_idx])
        except Exception:

            val_sd = float(A_weighted[s_idx, d_idx].toarray()[0, 0])
        try:
            val_ds = float(A_weighted[d_idx, s_idx])
        except Exception:
            val_ds = float(A_weighted[d_idx, s_idx].toarray()[0, 0])
        #max_val = max(val_sd, val_ds)
        #min_val = min(val_sd, val_ds)
        edge_importance[(src, dst)] = float(val_sd)
        edge_importance[(dst, src)] = float(val_ds)
        
        if float(A_new[s_idx, d_idx])<float(A_new[d_idx, s_idx]):
            node_importance[(src,dst)] = (float(A_new[s_idx, d_idx]),float(A_new[d_idx, s_idx]))
        else:
            node_importance[(dst,src)] = (float(A_new[d_idx, s_idx]),float(A_new[s_idx, d_idx]))


    #node_importance_idx = shano_sparse(A_new, A_sparse)
    #node_importance = {inv_map[idx]: score for idx, score in node_importance_idx.items()}

    return edge_importance, node_importance
