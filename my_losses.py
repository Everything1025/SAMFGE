import torch.nn.functional as F



def smoothness_loss(p, A_ones, device):
    A_torch = torch.FloatTensor(A_ones).to(device)
    diff_sq = (p.unsqueeze(1) - p.unsqueeze(0))**2
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



from sklearn.cluster import KMeans
import torch

def get_pseudo_labels_from_z_1(model, features, L_sym, L_rw, n_clusters):
    """
    通过模型提取融合特征 (z_fused) 并进行 KMeans 聚类生成伪标签
    """
    model.eval()
    with torch.no_grad():
        _, z_fused, _ = model(features, L_sym, L_rw)
    hidden_np = z_fused.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    return kmeans.fit_predict(hidden_np)

def pseudo_ce_loss(p, y_pseudo, eps=1e-8):
    """
    p: [N, K] 概率（p1/p2 或它们的平均）
    y_pseudo: [N]，KMeans 得到的伪标签（long 型，下标 0~K-1）
    """
    p = p.clamp(min=eps)
    log_p = torch.log(p)  # [N, K]
    loss = F.nll_loss(log_p, y_pseudo)
    return loss


def laplacian_smoothness(z, L_sym):
    """
    z: (N, F) dense
    L_sym: (N, N) torch sparse or dense (symmetric normalized Laplacian)
    returns: scalar = tr(z^T L z)
    """
    if L_sym.is_sparse:
        Lz = torch.sparse.mm(L_sym, z)
    else:
        Lz = L_sym @ z
    return torch.sum(z * Lz)


def w_low_order_prior(w):
    k = torch.arange(w.numel(), device=w.device, dtype=w.dtype)
    return torch.sum(k * w)


def graph_regularization_loss(Z, L):

    LZ = torch.matmul(L, Z)

    loss_graph = torch.trace(torch.matmul(Z.T, LZ))

    return loss_graph / Z.shape[0]
