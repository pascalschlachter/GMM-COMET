import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torch.cuda.amp import autocast

import numpy as np
import pandas as pd

import os

from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm

from networks import BaseModule, FeatureExtractor
from utils import HScore, GaussianMixtureModel, calculate_entropy, calculate_cosine_similarity, calculate_kld, mask, calculate_gmm_mahalanobis_distances
from augmentation import get_tta_transforms


class GmmBaAdaptationModule(BaseModule):
    def __init__(self, datamodule, feature_dim=256, lr=1e-2, red_feature_dim=64, p_reject=0.5, N_init=30,
                 augmentation=True, lam=1, temperature=0.1, alpha_GMM=1, ckpt_dir='', ood_metric='entropy_prediction',
                 use_mean_teacher=False, ensemble=False, alpha_MT=0.999, lam_consistency=1, lam_source_consistency=1):
        super(GmmBaAdaptationModule, self).__init__(datamodule, feature_dim, lr, ckpt_dir)

        self.ckpt_dir = ckpt_dir

        # ---------- Dataset information ----------
        self.source_class_num = datamodule.source_private_class_num + datamodule.shared_class_num
        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num

        # Additional feature reduction model
        self.feature_reduction = nn.Sequential(nn.Linear(feature_dim, red_feature_dim)).to(self.device)

        # L1 consistency loss
        self.consistency_loss = torch.nn.MSELoss()
        self.source_consistency_loss = torch.nn.MSELoss()

        # ---------- GMM ----------
        self.gmm = GaussianMixtureModel(self.source_class_num, alpha_GMM)

        # ---------- Unknown mask ----------
        self.ood_metric = ood_metric
        self.mask = mask(self, 0.5 - p_reject / 2, 0.5 + p_reject / 2, N_init, ood_metric)

        # ---------- Further initializations ----------
        self.tta_transform = get_tta_transforms()
        self.augmentation = augmentation
        self.temperature = temperature
        self.lam = lam

        # definition of h-score and accuracy
        self.total_online_tta_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        if self.open_flag:
            self.total_online_tta_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)

        self.domainwise_hscore = []
        self.domainwise_accuracy = []
        self.domain_idx = 0
        if isinstance(datamodule.test_domain, list):
            if self.open_flag:
                self.domainwise_hscore = [HScore(self.known_classes_num, datamodule.shared_class_num) for i in datamodule.test_domain]
            self.domainwise_accuracy = [Accuracy(task='multiclass', num_classes=self.known_classes_num + 1) for _ in datamodule.test_domain]

        self.automatic_optimization = False

        # Mean Teacher
        if use_mean_teacher == True:
            self.use_mean_teacher = True
            self.ensemble = ensemble
            self.alpha_MT = alpha_MT
            self.lam_consistency = lam_consistency
            self.backbone_teacher = self.copy_model(self.backbone)
            self.feature_extractor_teacher = self.copy_model(self.feature_extractor)
            self.classifier_teacher = self.copy_model(self.classifier)

            # Freeze teacher model (not trainable, eval mode)
            for p in self.backbone_teacher.parameters():
                p.requires_grad = False
            self.backbone_teacher.eval()

            for p in self.feature_extractor_teacher.parameters():
                p.requires_grad = False
            self.feature_extractor_teacher.eval()

            for p in self.classifier_teacher.parameters():
                p.requires_grad = False
            self.classifier_teacher.eval()
        else:
            self.use_mean_teacher = False
            self.ensemble = False

        self.lam_source_consistency = lam_source_consistency
        if self.lam_source_consistency > 0:
            self.source_backbone = self.copy_model(self.backbone)
            self.source_feature_extractor = self.copy_model(self.feature_extractor)
            self.source_classifier = self.copy_model(self.classifier)

    def configure_optimizers(self):
        # define different learning rates for different subnetworks
        params_group = []

        for k, v in self.backbone.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 0.1}]
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.feature_reduction.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]

        optimizer = torch.optim.SGD(params_group, momentum=0.9, nesterov=True)
        return optimizer

    # Mean Teacher Experiment
    def forward_teacher(self, x, apply_softmax=True):
        x = self.backbone_teacher(x)
        feature_embed = self.feature_extractor_teacher(x)
        x = self.classifier_teacher(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed
    
    def forward_source(self, x, apply_softmax=True):
        x = self.source_backbone(x)
        feature_embed = self.source_feature_extractor(x)
        x = self.source_classifier(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed

    # Mean Teacher Experiment
    def copy_model(self, model):
        if not isinstance(model, FeatureExtractor):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    # Mean Teacher
    def update_ema_variables(self, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model
    
    def on_train_start(self):
        for hscore in self.domainwise_hscore:
            hscore = hscore.to(self.device)
        for acc in self.domainwise_accuracy:
            acc = acc.to(self.device)

    def training_step(self, train_batch, batch_idx):
        # ----------- Open-World Test-time Training ------------
        self.backbone.eval()
        self.feature_extractor.eval()
        self.classifier.eval()

        opt = self.optimizers()
        opt.zero_grad()

        with autocast():
            x, y = train_batch

            # Determine ground truth for the ODA or OPDA scenario
            y = torch.where(y >= self.source_class_num, self.source_class_num, y)
            y = y.to(self.device)

            y_hat, feat_ext = self.forward(x)
            y_hat_aug, feat_ext_aug = self.forward(self.tta_transform(x))

            feat_ext_original = feat_ext

            # Mean Teacher
            if self.use_mean_teacher:
                y_hat_teacher, feat_ext_teacher = self.forward_teacher(x)

                feat_ext_teacher_original = feat_ext_teacher

            with torch.no_grad():
                feat_ext = self.feature_reduction(feat_ext)
                feat_ext_aug = self.feature_reduction(feat_ext_aug)
                # Update the GMM
                y_hat_clone_detached = y_hat.clone().detach()
                if self.use_mean_teacher:
                    feat_ext_teacher = self.feature_reduction(feat_ext_teacher)
                    y_hat_teacher_clone_detached = y_hat_teacher.clone().detach()
                    self.gmm.soft_update(feat_ext_teacher, y_hat_teacher_clone_detached)
                    max_values, pseudo_labels, likelihood = self.gmm.get_labels(feat_ext_teacher)
                else:
                    self.gmm.soft_update(feat_ext, y_hat_clone_detached)
                    max_values, pseudo_labels, likelihood = self.gmm.get_labels(feat_ext)

                pseudo_labels = pseudo_labels.to(self.device)
                likelihood = likelihood.to(self.device)

            # ---------- Generate a mask and monitor the result ----------
            y_hat_clone = y_hat.clone()
            if self.use_mean_teacher:
                known_mask, unknown_mask, rejection_mask, ood_values = self.mask.calculate_mask(likelihood, y_hat_teacher_clone_detached, feat_ext_teacher, self.gmm.mu, self.gmm.C)
            else:
                known_mask, unknown_mask, rejection_mask, ood_values = self.mask.calculate_mask(likelihood, y_hat_clone, feat_ext, self.gmm.mu, self.gmm.C)
            known_mask = known_mask.to(self.device)
            unknown_mask = unknown_mask.to(self.device)
            rejection_mask = rejection_mask.to(self.device)

            # Assign unknown pseudo-labels
            pseudo_labels[unknown_mask] = self.source_class_num

            # ---------- Enable OPDA for predictions ----------
            if self.ensemble:
                _, preds = torch.max((y_hat_clone_detached + y_hat_teacher_clone_detached) / 2, dim=1)
                if self.ood_metric == 'entropy_prediction':
                    ood_values = calculate_entropy((y_hat_clone + y_hat_teacher_clone_detached) / 2)
            else:
                _, preds = torch.max(y_hat_clone_detached, dim=1)
                if self.ood_metric == 'entropy_prediction' and self.use_mean_teacher == True:
                    ood_values = calculate_entropy(y_hat_clone)
            unknown_threshold = (self.mask.tau_low + self.mask.tau_high) / 2

            output_mask = torch.zeros_like(ood_values, dtype=torch.bool)
            output_mask[ood_values >= unknown_threshold] = True
            preds[output_mask] = self.source_class_num

            # ---------- Calculate the loss -----------
            # ---------- Contrastive loss -----------
            feat_ext = feat_ext.to(self.device)
            feat_ext_aug = feat_ext_aug.to(self.device)
            if self.augmentation:
                feat_total = torch.cat([feat_ext, feat_ext_aug], dim=0)
            else:
                feat_total = feat_ext
            mu = self.gmm.mu.to(self.device)
            # Calculate all cosine similarities between features (embeddings)
            cos_feat_feat = torch.exp(calculate_cosine_similarity(feat_total, feat_total) / self.temperature)
            # Calculate all cosine similarities between features (embeddings) and GMM means
            cos_feat_mu = torch.exp(calculate_cosine_similarity(mu, feat_total) / self.temperature)          

            if self.augmentation:
                known_mask_rep = known_mask.repeat(2)
                pseudo_labels_rep = pseudo_labels.repeat(2)
            else:
                known_mask_rep = known_mask
                pseudo_labels_rep = pseudo_labels

            divisor = torch.sum(cos_feat_mu, dim=1)
            logarithmus = torch.log(torch.divide(cos_feat_mu, divisor.unsqueeze(1)))
            used = torch.gather(logarithmus[known_mask_rep], 1, pseudo_labels_rep[known_mask_rep].view(-1, 1))
            L_mu_feat = torch.sum(torch.sum(used, dim=0))

            # Minimize distance between known features of the same class
            # Maximize distance between known/unknown features of different classes
            divisor = torch.sum(cos_feat_feat, dim=0)
            logarithmus = torch.log(torch.divide(cos_feat_feat, divisor.unsqueeze(0)))
            # Calculate the equality between elements of pseudo_label along both axes
            pseudo_label_rep_expanded = pseudo_labels_rep.unsqueeze(1)
            mask = pseudo_label_rep_expanded == pseudo_labels_rep
            used = torch.zeros_like(logarithmus)
            used[mask] = logarithmus[mask.bool()]
            L_feat_feat = torch.sum(torch.sum(used[known_mask_rep, known_mask_rep], dim=0))
            L_con = L_mu_feat + L_feat_feat

            # Entropy Loss
            y_hat_entropy = -torch.matmul(y_hat, torch.log2(y_hat.T)) / torch.log2(torch.tensor(self.known_classes_num))
            y_hat_entropy = torch.diagonal(y_hat_entropy)
            entropy_loss = y_hat_entropy[known_mask].mean() -\
                           y_hat_entropy[unknown_mask].mean()

            if self.lam_source_consistency > 0:
                _, feat_ext_source = self.forward_source(x)
                source_consistency_loss = self.source_consistency_loss(feat_ext_original, feat_ext_source)

            if self.use_mean_teacher:
                # consistency loss
                conistency_loss = self.consistency_loss(feat_ext_original, feat_ext_teacher_original)

                if self.lam_source_consistency > 0:
                    self.loss = L_con + self.lam * entropy_loss + self.lam_consistency * conistency_loss + self.lam_source_consistency * source_consistency_loss
                else:
                    self.loss = L_con + self.lam * entropy_loss + self.lam_consistency * conistency_loss
            else:
                if self.lam_source_consistency > 0:
                    self.loss = L_con + self.lam * entropy_loss + self.lam_source_consistency * source_consistency_loss
                else:
                    self.loss = L_con + self.lam * entropy_loss

            self.manual_backward(self.loss)
            opt.step()

        # log into progress bar
        self.log('train_loss', self.loss, on_epoch=True, prog_bar=True)
        # self.log('train_acc', self.total_train_acc, on_epoch=True, prog_bar=True)
        self.log('tta_acc', self.total_online_tta_acc, on_epoch=True, prog_bar=True)

        if self.use_mean_teacher:
            self.backbone_teacher = self.update_ema_variables(ema_model=self.backbone_teacher, model=self.backbone,
                                                            alpha_teacher=self.alpha_MT)
            self.feature_extractor_teacher = self.update_ema_variables(ema_model=self.feature_extractor_teacher,
                                                                    model=self.feature_extractor,
                                                                    alpha_teacher=self.alpha_MT)
            self.classifier_teacher = self.update_ema_variables(ema_model=self.classifier_teacher,
                                                                model=self.classifier,
                                                                alpha_teacher=self.alpha_MT)

        ### Prediction
        # ---------- Update the H-Score ----------
        self.total_online_tta_acc(preds, y)
        if self.open_flag:
            self.total_online_tta_hscore.update(preds, y)
        if self.domainwise_hscore != []:
            self.domainwise_hscore[self.domain_idx].update(preds, y)
        if self.domainwise_accuracy != []:
            self.domainwise_accuracy[self.domain_idx](preds, y)

        if isinstance(self.trainer.datamodule.test_domain, list):
            if batch_idx == self.trainer.datamodule.batches_per_domain_accu[self.domain_idx]-1:
                accuracy = self.domainwise_accuracy[self.domain_idx].compute()
                print(f"Accuracy (Domain: {self.trainer.datamodule.test_domain[self.domain_idx]}): {accuracy}")
                if self.open_flag:
                    h_score, known_acc, unknown_acc = self.domainwise_hscore[self.domain_idx].compute()
                    print(f"H-Score (Domain: {self.trainer.datamodule.test_domain[self.domain_idx]}): {h_score}")
                    print(f"Known Accuracy (Domain: {self.trainer.datamodule.test_domain[self.domain_idx]}): {known_acc}")
                    print(f"Unknown Accuracy (Domain: {self.trainer.datamodule.test_domain[self.domain_idx]}): {unknown_acc}")
                self.domain_idx += 1

        return self.loss

    def on_train_epoch_end(self):
        tau = torch.cat((torch.tensor([self.mask.tau_low], dtype=torch.float32), torch.tensor([self.mask.tau_high], dtype=torch.float32)))
        t_np = tau.numpy() #convert to Numpy array
        df = pd.DataFrame(t_np) #convert to a dataframe
        df.to_csv("tau",index=True) #save to file
        # ---------- Monitor the performance of the OPDA setting ----------
        print(f"Accuracy: {self.total_online_tta_acc.compute()}")
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.total_online_tta_hscore.compute()
            print(f"H-Score (Epoch): {h_score}")
            print(f"Known Accuracy (Epoch): {known_acc}")
            print(f"Unknown Accuracy: {unknown_acc}")
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('Epoch UnknownAcc', unknown_acc)

            print('\n H-Scores of all domains:')
            for hscore in self.domainwise_hscore:
                print(str(hscore.compute()[0].cpu().numpy() * 100).replace('.',','))
            
        print('\n \n Accuracies of all domains:')
        for acc in self.domainwise_accuracy:
            print(str(acc.compute().cpu().numpy() * 100).replace('.',','))
