import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss

logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class DHFN():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        # Extract model parameters
        params = model[0].parameters()

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DHFN = model[0]
        net.append(net_DHFN)
        model = net

        # Get and print ablation experiment configuration
        ablation_name = getattr(model[0], 'ablation_name', 'Full Model')
        logger.info(f"🔬 Training Started - Ablation Config: {ablation_name}")

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision)

                    # ==================== Get Ablation Configuration ====================
                    use_disentanglement = output.get('use_disentanglement', True)
                    use_hierarchical_learning = output.get('use_hierarchical_learning', True)
                    use_shared_enhance = output.get('use_shared_enhance', True)
                    ablation_name = output.get('ablation_name', 'Full Model')

                    # ==================== Task Loss (L_MSA) ====================
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)

                    # Total MSA loss L_MSA
                    # Adjust loss weights based on ablation configuration
                    weight_l_hetero = 3.0 if use_hierarchical_learning else 1.0
                    weight_c = 1.0  # Shared enhancement ablation doesn't affect loss_task_c weight

                    loss_task = 1 * loss_task_all + weight_c * loss_task_c + weight_l_hetero * loss_task_l_hetero

                    # ==================== Disentanglement-related Losses ====================
                    # Only computed when use_disentanglement=True
                    if use_disentanglement:
                        # Reconstruction loss L_r
                        loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                        loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                        loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                        loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                        # Cycle-consistency loss L_c
                        s_l_r_corrected = output['s_l_r'].permute(0, 2, 1)  # [B, T', d] -> [B, d, T']
                        loss_sl_slr = self.MSE(output['s_l'], s_l_r_corrected)
                        s_v_r_corrected = output['s_v_r'].permute(0, 2, 1)
                        loss_sv_slv = self.MSE(output['s_v'], s_v_r_corrected)
                        s_a_r_corrected = output['s_a_r'].permute(0, 2, 1)
                        loss_sa_sla = self.MSE(output['s_a'], s_a_r_corrected)
                        loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                        # Orthogonality loss L_o (cosine similarity minimization)
                        if self.args.dataset_name == 'mosi':
                            num = 50
                        elif self.args.dataset_name == 'mosei':
                            num = 10

                        cosine_similarity_s_c_l = self.cosine(
                            output['s_l'].reshape(-1, num),
                            output['c_l'].reshape(-1, num),
                            torch.tensor([-1]).cuda()
                        )
                        cosine_similarity_s_c_v = self.cosine(
                            output['s_v'].reshape(-1, num),
                            output['c_v'].reshape(-1, num),
                            torch.tensor([-1]).cuda()
                        )
                        cosine_similarity_s_c_a = self.cosine(
                            output['s_a'].reshape(-1, num),
                            output['c_a'].reshape(-1, num),
                            torch.tensor([-1]).cuda()
                        )
                        loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                        # Triplet margin loss L_m (cross-modal alignment)
                        c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                        ids, feats = [], []
                        for i in range(labels.size(0)):
                            feats.append(c_l[i].view(1, -1))
                            feats.append(c_v[i].view(1, -1))
                            feats.append(c_a[i].view(1, -1))
                            ids.append(labels[i].view(1, -1))
                            ids.append(labels[i].view(1, -1))
                            ids.append(labels[i].view(1, -1))
                        feats = torch.cat(feats, dim=0)
                        ids = torch.cat(ids, dim=0)
                        loss_sim = self.sim_loss(ids, feats)

                        # Overall loss L_DHFN (full version)
                        # L_DHFN = L_MSA + λ1(L_r + L_c + λ2(L_m + L_o))
                        combined_loss = loss_task + (loss_s_sr + loss_recon + (loss_sim + loss_ort) * 0.1) * 0.1
                    else:
                        # ==================== Ablation Version: Without Disentanglement ====================
                        # Only use task loss
                        combined_loss = loss_task

                    combined_loss.backward()

                    # Gradient clipping
                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # Validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            # Save checkpoint for each epoch
            torch.save(model[0].state_dict(), './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')

            # Save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/DHFN' + str(self.args.dataset_name) + '.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # Early stopping
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results