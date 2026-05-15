import argparse
import os
import torch.nn.functional as F
from torch import optim
from utils import *
from my_model import FAEEL_Module, recalculate_laplacians
from functions import get_data, normalize, spixel_to_pixel_labels, cluster_accuracy, run_superpixel_segmentation, \
    get_model_class
from my_losses import smoothness_loss, confidence_loss,cluster_balance_loss,get_pseudo_labels_from_z,pseudo_ce_loss, \
    graph_regularization_loss

device = torch.device("cpu")

if __name__ == '__main__':
    GLOBAL_SEED = 42
    setup_seed(GLOBAL_SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='my_model', help='Model architecture: ')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--dataset', type=str, default='PaviaU', help='type of dataset.')
    parser.add_argument('--superpixel_scale', type=int, default=100, help="superpixel_scale")
    parser.add_argument('--lr_edge', type=float, default=1e-3, help='Learning rate for FA-EEL module.')

    args = parser.parse_args()
    datasets_list = ['PaviaU',]

    scale_config = {
        'PaviaU': 600,# k=4
    }

    hyperparam_config = {
        'PaviaU': {
            'lambda_mse': 1.0, 'lambda_smooth': 1.0, 'lambda_conf': 1.0,
            'lambda_balance': 1.0, 'lambda_pseudo':3.0, 'lambda_graph':0.1
        },

        'default': {
            'lambda_mse': 1.0, 'lambda_smooth': 1.0, 'lambda_conf': 1.0,
            'lambda_balance': 0.8,'lambda_pseudo':1.0
        }
    }

    # ================= 3. 设置结果文件保存路径（当前目录） =================
    current_dir = os.getcwd()

    print(f"Plan to run datasets: {datasets_list}")

    # ================= 4. 开始遍历数据集 =================
    for dataset_name in datasets_list:
        args.dataset = dataset_name

        if dataset_name in scale_config:
            args.superpixel_scale = scale_config[dataset_name]
        else:
            print(f"Warning: No scale config found for {dataset_name}, using default: {args.superpixel_scale}")

        print(f"\n{'=' * 40}")
        print(f"Processing: {args.dataset} | Scale: {args.superpixel_scale}")
        print(f"{'=' * 40}")

        # ----------------- 数据加载与预处理 -----------------
        try:
            input, num_classes, y_true, gt_reshape, gt_hsi = get_data(args.dataset)
        except Exception as e:
            print(f"Error loading dataset {args.dataset}: {e}")
            continue
        else:
            labeled_indices = np.where(gt_reshape != 0)[0]
            total_labeled = labeled_indices.shape[0]

        input_normalize = normalize(input)
        input_numpy = np.array(input_normalize)

        try:
            Q, S, A, Edge_index, Edge_atter, Seg, A_ones = run_superpixel_segmentation(
                dataset_name=args.dataset,
                input_numpy=input_numpy,
                gt_hsi=gt_hsi,
                num_classes=num_classes,
                scale=args.superpixel_scale
            )
        except ValueError as e:
            print(f"Superpixel segmentation failed for {args.dataset}: {e}")
            continue

        true_labels = gt_reshape
        adj = sp.csr_matrix(A_ones)

        args.cluster_num = num_classes

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

        L_sym_np = preprocess_graph(adj, norm='sym').toarray()
        L_sym = torch.from_numpy(L_sym_np).float().to(device)

        L_rw_np = preprocess_graph(adj, norm='left').toarray()
        L_rw = torch.from_numpy(L_rw_np).float().to(device)

        features = torch.from_numpy(S).float().to(device)

        current_params = hyperparam_config.get(dataset_name, hyperparam_config['default'])

        lambda_mse = current_params['lambda_mse']
        lambda_smooth = current_params['lambda_smooth']
        lambda_conf = current_params['lambda_conf']
        lambda_balance = current_params['lambda_balance']
        lambda_pseudo = current_params['lambda_pseudo']
        lambda_graph = current_params['lambda_graph']
        warmup_epochs = 0
        kmeans_epoch = 10
        pseudo_labels_torch = None

        num_runs = 1
        ModelClass = get_model_class(args.model_name)

        for run_idx in range(num_runs):

            model = ModelClass(
                in_dim=features.shape[1],
                num_clusters=num_classes,
            ).to(device)

            faeel = FAEEL_Module(in_dim=features.shape[1]).to(device)

            optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': args.lr},
                {'params': faeel.parameters(), 'lr': args.lr_edge}
            ])

            model = model.to(device)

            current_adj = torch.from_numpy(adj.toarray()).float().to(device)
            target = current_adj + torch.eye(current_adj.shape[0], device=device)

            current_L_sym = L_sym.clone()
            current_L_rw = L_rw.clone()

            gamma = 0.8
            lambda_edge = 0.2
            edge_warmup_epochs = 20

            print(f'[{args.dataset}] Run {run_idx + 1}/{num_runs}   ...')

            # ================= 新增：初始化记录最优指标的变量 =================
            best_epoch = -1
            best_acc = -1.0
            best_metrics = {}

            for epoch in range(args.epochs):
                model.train()
                faeel.train()
                q, z, z_low_c, z_high_c = model(features, current_L_sym, current_L_rw)
                optimizer.zero_grad()

                w_pre_masked, w_emp_masked = faeel(z, z_low_c, z_high_c, q, current_adj)

                z_norm = F.normalize(z, p=2, dim=1)
                sim = z_norm @ z_norm.T

                loss_mse = F.mse_loss(sim, target)

                loss_smooth = smoothness_loss(q, target, device)
                loss_conf = confidence_loss(q)
                loss_balance = cluster_balance_loss(q)
                loss_graph_reg = graph_regularization_loss(z, L_sym)

                loss_pseudo = 0.0

                if epoch >= warmup_epochs:
                    if pseudo_labels_torch is None and epoch % kmeans_epoch == 0:
                        labels_np = get_pseudo_labels_from_z(model, features, L_sym, L_rw, num_classes)
                        pseudo_labels_torch = torch.from_numpy(labels_np).long().to(device)

                    if pseudo_labels_torch is not None:
                        loss_pseudo = pseudo_ce_loss(q, pseudo_labels_torch)

                mask = current_adj > 0
                loss_edge = F.mse_loss(w_pre_masked[mask], w_emp_masked[mask])

                current_lambda_edge = lambda_edge

                loss = (
                        lambda_mse * loss_mse +
                        lambda_conf * loss_conf
                        + lambda_balance * loss_balance
                        + lambda_smooth * loss_smooth
                        + lambda_pseudo * loss_pseudo
                        + lambda_graph * loss_graph_reg
                        + current_lambda_edge * loss_edge
                )
                loss.backward()
                optimizer.step()

                if epoch % edge_warmup_epochs == 0:
                    with torch.no_grad():
                        new_adj = gamma * current_adj + (1 - gamma) * w_pre_masked.detach()
                        new_adj[new_adj < 0.1] = 0
                        current_adj = new_adj
                        current_L_sym, current_L_rw = recalculate_laplacians(current_adj)
                        target = current_adj + torch.eye(current_adj.shape[0], device=device)


                if epoch % 1 == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_q, _, _, _ = model(features, current_L_sym, current_L_rw)
                        predict_labels_sp = torch.argmax(eval_q, dim=1).cpu().numpy()  # 超像素级标签（0~K-1）

                        indx = np.where(gt_reshape != 0)
                        labels = gt_reshape[indx]
                        pixel_y = spixel_to_pixel_labels(predict_labels_sp, Q)
                        prediction = pixel_y[indx]

                        acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels, prediction, return_aligned=False)
                        print(f'[Epoch: {epoch:03d}] ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}'
                              f'  kappa: {kappa:.4f}, pur: {pur:.4f}')


                        if acc > best_acc:
                            best_acc = acc
                            best_epoch = epoch
                            best_metrics = {
                                'acc': acc, 'nmi': nmi, 'ari': ari,
                                'kappa': kappa, 'pur': pur, 'ca': ca
                            }


            print(f"\n{'-' * 50}")
            print(f"Best ACC   : {best_metrics['acc']:.4f}")
            print(f"NMI        : {best_metrics['nmi']:.4f}")
            print(f"ARI        : {best_metrics['ari']:.4f}")
            print(f"Kappa      : {best_metrics['kappa']:.4f}")
            print(f"Purity     : {best_metrics['pur']:.4f}")
            print(f"{'-' * 50}\n")