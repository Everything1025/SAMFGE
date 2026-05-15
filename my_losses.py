import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def smoothness_loss(p, A_ones, device):
    A_torch = torch.FloatTensor(A_ones).to(device)
    diff_sq = (p.unsqueeze(1) - p.unsqueeze(0)) ** 2
    loss_smooth = (A_torch * diff_sq.sum(dim=2)).sum() / (A_torch.sum() + 1e-8)
    return loss_smooth


def confidence_loss(p):
    return -(p.clamp(min=1e-8) * torch.log(p.clamp(min=1e-8))).sum(dim=1).mean()


def cluster_balance_loss(p):
    q = p.mean(dim=0).clamp(min=1e-8)
    return (q * torch.log(q)).sum()


def get_pseudo_labels_from_z(model, features, L_sym, L_rw, n_clusters):
    model.eval()
    with torch.no_grad():
        _, z_fused, _, _ = model(features, L_sym, L_rw)
    hidden_np = z_fused.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    return kmeans.fit_predict(hidden_np)


def pseudo_ce_loss(p, y_pseudo, eps=1e-8):
    p = p.clamp(min=eps)
    log_p = torch.log(p)
    loss = F.nll_loss(log_p, y_pseudo)
    return loss


def graph_regularization_loss(Z, L):
    LZ = torch.matmul(L, Z)
    loss_graph = torch.trace(torch.matmul(Z.T, LZ))
    return loss_graph / Z.shape[0]
