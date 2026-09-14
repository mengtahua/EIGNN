import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GCN2Conv, SGConv
from torch_geometric.utils import to_undirected
import random
import matplotlib.pyplot as plt
import time

# ---------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------

def preprocess_dataset(dataset):
    x = torch.tensor(dataset['x'], dtype=torch.float)
    y = torch.tensor(dataset['label'], dtype=torch.long)
    edge_index = np.array(dataset['edge_index'].T, dtype=np.int64)
    edge_index = to_undirected(torch.tensor(edge_index.T, dtype=torch.long))
    edge_importance = dataset['edge_importance']  # dict {(i,j): weight}
    print(f"Edge importance length: {len(edge_importance)}")
    return x, y, edge_index, edge_importance

# ---------------------------

def analyze_distribution(
    edge_index, 
    edge_importance, 
    mask=None, 
    title="Edge Importance Distribution",
    use_log=True,               
    linewidth=2,
):
    """
    
    - kept color: #ffc75f
    - removed color: #4d8076
    - legend placed at upper right
    """

    # ------------------------
    
    # ------------------------
    edges = edge_index.cpu().numpy().T  # shape (E,2)
    imp_values = []
    for i, j in edges:
       
        if isinstance(edge_importance, dict):
            w = edge_importance.get((i, j), edge_importance.get((j, i), 1.0))
        else:
            
            
            try:
                w = float(edge_importance[len(imp_values)])
            except Exception:
                w = 1.0
        imp_values.append(w)
    imp_values = np.array(imp_values)

    # ------------------------

    if mask is not None:
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        
        if mask_np.shape[0] != imp_values.shape[0]:
            raise ValueError(f"Mask length ({mask_np.shape[0]}) != number of edges ({imp_values.shape[0]})")

        kept = imp_values[mask_np.astype(bool)]
        removed = imp_values[~mask_np.astype(bool)]

        print(f"[{title}] kept={len(kept)}, removed={len(removed)}")
        if len(kept) > 0:
            print(f"  Kept: mean={kept.mean():.4f}, median={np.median(kept):.4f}, P90={np.percentile(kept,90):.4f}")
        if len(removed) > 0:
            print(f"  Removed: mean={removed.mean():.4f}, median={np.median(removed):.4f}")

    return kept.mean(),np.median(kept),removed.mean(),np.median(removed) #analyze_distribution

class MyGINConv(nn.Module):
    def __init__(self, nn_mlp, eps=0.0, train_eps=False, dropout=0.5):
        super(MyGINConv, self).__init__()
        self.nn = nn_mlp
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(nn_mlp[-1].out_features)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float))
        else:
            self.register_buffer('eps', torch.tensor(eps, dtype=torch.float))

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        device = x.device
        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=device)
        else:
            edge_weight = edge_weight.to(device)

        agg = torch.zeros_like(x)
        agg.index_add_(0, col, x[row] * edge_weight.unsqueeze(1))

        out = (1 + self.eps) * x + agg
        out = self.nn(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class MySAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, normalize=False):
        super(MySAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)   # self
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)  # neighbor

        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        device = x.device

        # ---------------------------
       
        # ---------------------------
        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=device)
        else:
            edge_weight = edge_weight.to(device)
        # ---------------------------
        
        # ---------------------------
        agg = torch.zeros_like(x)

        # sum_j w_ij * x_j
        agg.index_add_(0, col, x[row] * edge_weight.unsqueeze(1))

        if self.normalize:
            # degree = sum_j w_ij
            deg = torch.zeros(x.size(0), device=device)
            deg.index_add_(0, col, edge_weight)
            deg = deg.clamp(min=1).unsqueeze(1)

            agg = agg / deg  # mean aggregation

        # ---------------------------
        
        # h_i = W1 * x_i + W2 * agg_i
        # ---------------------------
        out = self.lin_l(x) + self.lin_r(agg)

        # ---------------------------
        
        # ---------------------------
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out
# ======================
# STE topk 
# ======================
class STE_topk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):

        topk_vals, topk_idx = torch.topk(scores, k)
        mask = torch.zeros_like(scores)
        mask[topk_idx] = 1.0
        ctx.save_for_backward(scores, mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, mask = ctx.saved_tensors

        grad_input = grad_output.clone()
        return grad_input, None

def ste_topk(scores, k):

    return STE_topk.apply(scores, k)

# ======================

class EIGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 edge_index, edge_importance, conv_type='GCN',
                 min_keep_ratio=0.8, max_keep_ratio=0.9, device='cpu', tau=1.0):
        super().__init__()

        self.device = device
        self.tau = tau   
        self.conv_type = conv_type
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.num_edges = edge_index.size(1)

        self.register_buffer("edge_index", edge_index)

        # ---------------------------

        edges = edge_index.cpu().numpy().T
        feats = []
        w_vals = []
        for i, j in edges:
            w = edge_importance.get((i, j), edge_importance.get((j, i), 1.0))
            w_vals.append(w)
            feats.append([w, w**0.5, w**1.5, w**2.5, w**2, w**3])

        feats_tensor = torch.tensor(feats, dtype=torch.float)
        min_vals = feats_tensor.min(dim=0, keepdim=True)[0]
        max_vals = feats_tensor.max(dim=0, keepdim=True)[0]
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        feats_tensor = (feats_tensor - min_vals) / range_vals
        self.register_buffer("edge_feat", feats_tensor)

        w_tensor = torch.tensor(w_vals, dtype=torch.float)
        w_min = w_tensor.min()
        w_max = w_tensor.max()
        if w_max > w_min:
            fixed_weight = (w_tensor - w_min) / (w_max - w_min)
        else:
            fixed_weight = torch.ones_like(w_tensor)
        self.register_buffer("fixed_weight", fixed_weight)

        # ---------------------------
        # edge MLP
        # ---------------------------
        self.edge_mlp = nn.Sequential(
            nn.Linear(6, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

        self.in_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            #nn.Linear(hidden_channels*2, hidden_channels),
            #nn.ReLU(),
        )

        # ---------------------------
        # GNN layers
        # ---------------------------
        if conv_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_type == 'GraphSAGE':
            self.conv1 = MySAGEConv(in_channels, hidden_channels)
            self.conv2 = MySAGEConv(hidden_channels, out_channels)
        elif conv_type == 'GIN':
            nn1 = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            nn2 = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )
            self.conv1 = MyGINConv(nn1)
            self.conv2 = MyGINConv(nn2)
            #self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_type == 'GCN2':
            self.conv1 = GCN2Conv(hidden_channels, alpha=0.1, theta=0.5, layer=1)
            self.conv2 = GCN2Conv(hidden_channels, alpha=0.1, theta=0.5, layer=2)
            self.init_lin = nn.Linear(in_channels, hidden_channels)
            self.out_lin = nn.Linear(hidden_channels, out_channels)
        elif conv_type == 'SGC':
            self.conv1 = SGConv(in_channels, hidden_channels, K=2)
            self.conv2 = SGConv(hidden_channels, out_channels, K=2)
        else:
            raise NotImplementedError

    # ---------------------------

    def forward(self, x, test_mode=False):
        x = x.clamp(-10,10)

        if self.conv_type == 'GCN2':
            x0 = F.relu(self.init_lin(x))
            x1 = F.relu(self.conv1(x0, x0, self.edge_index))
        else:
            x1 = F.relu(self.conv1(x, self.edge_index))

        # edge scoring & selection
        if test_mode:

            with torch.no_grad():
                edge_logits = self.edge_mlp(self.edge_feat).squeeze()
                edge_logits = torch.nan_to_num(edge_logits, nan=0.0, posinf=1.0, neginf=0.0)
                edge_logits = edge_logits.clamp(-10,10)
                edge_probs = torch.sigmoid(edge_logits).clamp(1e-6,1-1e-6)

                min_keep = int(self.min_keep_ratio * self.num_edges)
                max_keep = int(self.max_keep_ratio * self.num_edges)
                mask = torch.bernoulli(edge_probs).bool()
                num_keep = mask.sum().item()
                if num_keep < min_keep:
                    missing = min_keep - num_keep
                    not_selected = (~mask).nonzero(as_tuple=True)[0]
                    if not_selected.numel() > 0:
                        topk_missing = torch.topk(edge_probs[not_selected], min(missing, not_selected.numel())).indices
                        mask[not_selected[topk_missing]] = True
                elif num_keep > max_keep:
                    selected = mask.nonzero(as_tuple=True)[0]
                    topk_keep = torch.topk(edge_probs[selected], max_keep).indices
                    new_mask = torch.zeros_like(mask)
                    new_mask[selected[topk_keep]] = True
                    mask = new_mask
                final_mask = mask
                used_edge_index = self.edge_index[:, final_mask]
                edge_weight = None
        else:
            edge_logits = self.edge_mlp(self.edge_feat).squeeze()
            edge_logits = torch.nan_to_num(edge_logits, nan=0.0, posinf=1.0, neginf=0.0)
            edge_logits = edge_logits.clamp(-10,10)

            edge_probs = torch.sigmoid(edge_logits).clamp(1e-6,1-1e-6)
            #print(edge_probs[:5])   # 可选打印


            mask_soft = ste_topk(edge_probs, int(self.max_keep_ratio * self.num_edges))
            mask_soft = torch.nan_to_num(mask_soft, nan=0.0)

            used_edge_index = self.edge_index
            edge_weight = mask_soft          
            
            final_mask = mask_soft > 0


        x1 = F.relu(self.in_lin(x1))
        if self.conv_type == 'GCN2':
            x1=self.in_lin(x1)
            x_out = F.relu(self.conv2(x1, x0, used_edge_index, edge_weight=edge_weight))
            x_out = self.out_lin(x_out)            
        else:
            x1=self.in_lin(x1)
            x_out = self.conv2(x1, used_edge_index, edge_weight=edge_weight)


        return x_out, used_edge_index, edge_probs, final_mask

# ---------------------------

def train_model(model, x, y, args, train_idx,eval_idx,test_idx,
                device='cpu'):
    x = x.to(device).clamp(-10,10)
    y = y.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    best_eval_acc, best_test_acc, best_state, best_mask = 0,0,None,None
    for epoch in range(args['epoch']):
        model.train()
        optimizer.zero_grad()
        out, used_edge_index, edge_probs, mask = model(x, test_mode=False)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():    n = x.size(0)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_ratio = args['train_ratio']
    cut1 = int(train_ratio * n)
    remaining = 1 - train_ratio
    cut2 = int((train_ratio + remaining / 2) * n)
    train_idx = torch.tensor(idx[:cut1], dtype=torch.long, device=device)
    eval_idx = torch.tensor(idx[cut1:cut2], dtype=torch.long, device=device)
    test_idx = torch.tensor(idx[cut2:], dtype=torch.long, device=device)
            out_eval, used_edge_index_eval,  _, mask_eval = model(x, test_mode=True)
            eval_acc = (out_eval[eval_idx].argmax(1)==y[eval_idx]).float().mean().item()
            if eval_acc > best_eval_acc:
                with torch.no_grad():
                    test_acc = (out_eval[test_idx].argmax(1)==y[test_idx]).float().mean().item()
                    best_test_acc = test_acc
                    best_eval_acc = eval_acc
                    best_state = model.state_dict()

                    #_, _, _, mask_eval = model(x, test_mode=True)
                    best_mask = mask_eval.clone()

                    

        if epoch % 100 == 0:
            print(f"Stage2 Epoch {epoch:03d} | Test Acc: {test_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_test_acc, best_mask

def run_experiment(dataset, args, conv_type='GCN', device=None):
    set_seed(args['seed'])
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y, edge_index, edge_importance = preprocess_dataset(dataset)
    x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)

    n = x.size(0)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_ratio = args['train_ratio']
    cut1 = int(train_ratio * n)
    remaining = 1 - train_ratio
    cut2 = int((train_ratio + remaining / 2) * n)
    train_idx = torch.tensor(idx[:cut1], dtype=torch.long, device=device)
    eval_idx = torch.tensor(idx[cut1:cut2], dtype=torch.long, device=device)
    test_idx = torch.tensor(idx[cut2:], dtype=torch.long, device=device)

    model = EIGNN(x.size(1), args['hidden'], int(y.max().item())+1,
                    edge_index, edge_importance, conv_type=conv_type, device=device).to(device)

    model, test_acc, mask = train_model(model, x, y, args, train_idx, eval_idx,test_idx, device=device)
    #print(mask[:5])

    print(args['seed'],conv_type,f"Final Test Accuracy: {test_acc:.4f} ")

    kept_mean,kept_median,removed_mean,removed_median = analyze_distribution(edge_index, edge_importance, mask)

    return test_acc,kept_mean,kept_median,removed_mean,removed_median
