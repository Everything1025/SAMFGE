import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedHSIClusteringModel(nn.Module):
    def __init__(self, in_dim, num_clusters, K=4,fusion_type='attention'):
        super(UnifiedHSIClusteringModel, self).__init__()
        self.K = K

        self.fusion_type = fusion_type


        self.theta_low = nn.Parameter(torch.FloatTensor(K + 1))

        self.theta_wide = nn.Parameter(torch.FloatTensor(K + 1))
        self.theta_narrow = nn.Parameter(torch.FloatTensor(K + 1))



        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim ),
            nn.ReLU(),
            nn.Linear(in_dim , in_dim),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Linear(in_dim * 2, 1),
            nn.Sigmoid()
        )

        self.alpha_logit = nn.Parameter(torch.tensor(0.0))


        self.cluster_head = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.ReLU(),
            nn.Linear(4 * in_dim, num_clusters),
            nn.ELU(),
            nn.Softmax(dim=1)
        )
        self.reset_parameters()

    def reset_parameters(self):


        with torch.no_grad():
            self.theta_low.fill_(0.0)
            self.theta_low[0] = 1.0
            self.theta_low[1] = -0.1



        nn.init.uniform_(self.theta_wide, 0.5,1.0)
        nn.init.uniform_(self.theta_narrow, 0.0, 0.5)
    def cheby_op(self, x, L):

        num_nodes = L.shape[0]
        I = torch.eye(num_nodes).to(L.device)
        L_hat = L - I
        T = [x]
        if self.K > 0:
            T.append(torch.mm(L_hat, x))
        for k in range(2, self.K + 1):
            T.append(2 * torch.mm(L_hat, T[-1]) - T[-2])
        return T


    def forward(self, x, L_sym, L_rw):

        T_low = self.cheby_op(x, L_sym)
        z_low = sum(self.theta_low[k] * T_low[k] for k in range(self.K + 1))

        T_rw = self.cheby_op(x, L_rw)

        z_high = sum(self.theta_wide[k] * T_rw[k] for k in range(self.K + 1)) - \
                 sum(self.theta_narrow[k] * T_rw[k] for k in range(self.K + 1))

        if self.fusion_type == "attention":

            c_weight = self.channel_gate(torch.mean(z_low + z_high, dim=0))
            z_low_c, z_high_c = z_low * c_weight, z_high * c_weight
            s_weight = self.spatial_gate(torch.cat([z_low_c, z_high_c], dim=1))
            z_fused = s_weight * z_low_c + (1 - s_weight) * z_high_c

        elif self.fusion_type == "global_scalar":

            alpha = torch.sigmoid(self.alpha_logit)
            z_fused = alpha * z_low + (1.0 - alpha) * z_high


        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        q = self.cluster_head(z_fused)
        return q, z_fused, z_low, z_high



class FAEEL_Module(nn.Module):
        def __init__(self, in_dim, l1=1.0, l2=1.0, l3=1.0):
            super(FAEEL_Module, self).__init__()

            self.W_edge = nn.Parameter(torch.FloatTensor(in_dim, in_dim))
            nn.init.xavier_uniform_(self.W_edge)


            self.l1 = l1
            self.l2 = l2
            self.l3 = l3

        def forward(self, z_fused, z_low_c, z_high_c, q, current_adj):
            edge_mask = current_adj > 0


            z_proj = torch.mm(z_fused, self.W_edge)
            w_pre_matrix = torch.sigmoid(torch.mm(z_proj, z_fused.t()))


            w_pre_matrix = (w_pre_matrix + w_pre_matrix.t()) / 2.0


            with torch.no_grad():


                z_low_norm = F.normalize(z_low_c, p=2, dim=1)
                sim_low = torch.mm(z_low_norm, z_low_norm.t())


                high_energy = torch.norm(z_high_c, p=2, dim=1, keepdim=True)  # [N, 1]


                penalty_matrix = high_energy + high_energy.t()


                penalty_masked = penalty_matrix * edge_mask.float()
                local_max = penalty_masked.max()
                dist_high_norm = penalty_masked / (local_max + 1e-8)


                q_norm = F.normalize(q, p=2, dim=1)
                sim_q = torch.mm(q_norm, q_norm.t())

                tau = 0.6
                evidence = (self.l1 * sim_low - self.l2 * dist_high_norm + self.l3 * sim_q) / tau
                w_emp_matrix = torch.sigmoid(evidence)




            w_pre_masked = w_pre_matrix * edge_mask.float()
            w_emp_masked = w_emp_matrix * edge_mask.float()

            w_pre_masked.fill_diagonal_(1.0)
            w_emp_masked.fill_diagonal_(1.0)

            return w_pre_masked, w_emp_masked

def recalculate_laplacians(adj_matrix):

        N = adj_matrix.shape[0]
        I = torch.eye(N, device=adj_matrix.device)

        A_hat = adj_matrix + I
        D_hat = A_hat.sum(dim=1)

        D_inv = torch.pow(D_hat + 1e-8, -1.0)
        D_inv_sqrt = torch.pow(D_hat + 1e-8, -0.5)

        L_rw = I - D_inv.unsqueeze(1) * A_hat
        L_sym = I - D_inv_sqrt.unsqueeze(1) * A_hat * D_inv_sqrt.unsqueeze(0)

        return L_sym, L_rw




