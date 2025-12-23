import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import math
from scipy.stats import multivariate_normal


def calculate_entropy(likelihood):
    entropy_values = -(likelihood * torch.log2(likelihood + 1e-10)).sum(dim=1)
    scale_factor = torch.log2(torch.tensor(likelihood.shape[1]))
    entropy_values = entropy_values / scale_factor
    return entropy_values


def calculate_kld(likelihood, true_dist):
    T = 0.1
    dividend = torch.sum(torch.exp(likelihood / T), dim=1)
    logarithmus = - torch.log(dividend)
    divisor = torch.sum(true_dist, dim=1)
    kld_values = - (1 / likelihood.shape[1]) * divisor * logarithmus
    return kld_values


def calculate_cosine_similarity(mu, feat):
    cosine_sim = F.cosine_similarity(mu.unsqueeze(0), feat.unsqueeze(1), dim=2)
    return cosine_sim


def calculate_euclidean_distance(mu, feat):
    # Calculate squared Euclidean distances
    euclidean_dist_sq = torch.sum((mu.unsqueeze(0) - feat.unsqueeze(1)) ** 2, dim=2) # (batch_size, source_class_num)

    # Take square root to get Euclidean distance
    euclidean_dist = torch.sqrt(euclidean_dist_sq)

    return euclidean_dist


def calculate_gmm_mahalanobis_distances(feat, mu, C):
    batch_size = feat.shape[0]
    n_modes = mu.shape[0]
    
    # Prepare a tensor to store distances
    distances = torch.zeros(batch_size, n_modes)
    
    # For each mode/component
    for n_mode in range(n_modes):
        mean = mu[n_mode]
        cov = C[n_mode]

        # Compute the inverse of the covariance matrix
        cov_inv = torch.linalg.inv(cov)
        
        # Expand the mean and covariance matrices to match the batch size
        mean_expanded = mean.unsqueeze(0).expand(batch_size, -1)
        cov_inv_expanded = cov_inv.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute differences
        diffs = feat - mean_expanded
        
        # Compute Mahalanobis distance using batch operations
        # Compute (x - mu)^T * cov_inv * (x - mu) for each sample
        distances[:, n_mode] = torch.sqrt(
            torch.bmm(diffs.unsqueeze(1), torch.bmm(cov_inv_expanded, diffs.unsqueeze(2))).squeeze()
        ).squeeze()
    
    return distances


def calculate_grad_norm(model, tensor, likelihood, T=1.0, epsilon=1e-6):
    grad_list = []

    # Ensure the model is in evaluation mode
    model.eval()

    # Iterate over each sample in the batch
    for i in range(likelihood.shape[0]):
        # Compute the KL term
        dividend = torch.sum(torch.exp(likelihood[i, :] / T))
        logarithmus = -torch.log(dividend + epsilon)
        divisor = torch.sum(tensor[i, :])
        kl = - (1 / likelihood.shape[1]) * divisor * logarithmus

        # Zero gradients before backward pass
        model.zero_grad()

        # Compute the gradients
        grads = torch.autograd.grad(kl, [model.classifier.fc.weight_g, model.classifier.fc.weight_v], retain_graph=True, allow_unused=True)

        if all(grad is not None for grad in grads):
            # Calculate the L1 norm of the gradients for both weight_g and weight_v
            grad_g = grads[0]
            grad_v = grads[1]

            l1_grad_g = torch.norm(grad_g, p=1).item() if grad_g is not None else 0.0
            l1_grad_v = torch.norm(grad_v, p=1).item() if grad_v is not None else 0.0

            # Combine the norms
            combined_grad_norm = l1_grad_g + l1_grad_v
            grad_list.append(combined_grad_norm)
        else:
            # Handle cases where gradients are None
            grad_list.append(0.0)

    # Convert the list of gradient norms to a tensor
    grad_norms_tensor = torch.tensor(grad_list, device=model.device)

    return grad_norms_tensor


class mask():
    def __init__(self, model, lower_percentage_threshold, upper_percentage_threshold, N_init, ood_metric):
        self.lower_percentage_threshold = lower_percentage_threshold
        self.upper_percentage_threshold = upper_percentage_threshold

        self.ood_metric = ood_metric
        self.model = model

        self.tau_low = None
        self.tau_low_list = []
        self.tau_high = None
        self.tau_high_list = []

        self.count = 0
        self.N_init = N_init

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_mask(self, likelihood, y_hat, feat, mu, C):
        if self.ood_metric == 'entropy_likelihood':
            ood_values = calculate_entropy(likelihood)
        elif self.ood_metric == 'entropy_prediction':
            ood_values = calculate_entropy(y_hat)
        elif self.ood_metric == 'mahalanobis_distance':
            ood_values, _ = torch.min(calculate_gmm_mahalanobis_distances(feat, mu, C), dim=1)
        elif self.ood_metric == 'euclidean_distance':
            ood_values, _ = torch.min(calculate_euclidean_distance(mu, feat), dim=1)
        elif self.ood_metric == 'grad_norm':
            new_dim = 5
            print("new_dim: ", new_dim)

            # likelihood_new = torch.empty(likelihood.shape[0], new_dim)
            _, indices = torch.sort(y_hat, dim=1, descending=True)
            top_values = y_hat.gather(1, indices[:, :new_dim])
            likelihood_new = top_values

            # likelihood_new = y_hat
            true_dist = torch.ones_like(likelihood_new) / likelihood_new.shape[1]
            ood_values = calculate_grad_norm(self.model, true_dist, likelihood_new)
        elif self.ood_metric == 'maximum_likelihood':
            ml, _ = torch.max(likelihood, dim=1)
            ood_values = 1- ml
        elif self.ood_metric == 'maximum_prediction':
            mp, _ = torch.max(y_hat, dim=1)
            ood_values = 1- mp
        else:
            raise NotImplementedError

        if self.count < self.N_init:
            # Sort values (from small to big)
            sorted_A, _ = torch.sort(ood_values)
            threshold_idx_known = math.ceil(len(sorted_A) * (self.lower_percentage_threshold))
            threshold_a = sorted_A[threshold_idx_known]
            self.tau_low_list.append(threshold_a)
            tau_low = torch.tensor(self.tau_low_list)
            threshold_idx_unknown = math.floor(len(sorted_A) * (self.upper_percentage_threshold))
            threshold_b = sorted_A[threshold_idx_unknown]
            self.tau_high_list.append(threshold_b)
            tau_high = torch.tensor(self.tau_high_list)
            self.tau_low = torch.mean(tau_low)
            self.tau_high = torch.mean(tau_high)

            self.count = self.count + 1

        # Determine the threshold value for the percentage_threshold values
        known_mask = torch.zeros_like(ood_values, dtype=torch.bool)
        known_mask[ood_values <= self.tau_low] = True
        tau_low = self.tau_low
        while torch.sum(known_mask).item() <= 1:
            tau_low += self.tau_low
            known_mask = ood_values < tau_low

        unknown_mask = torch.zeros_like(ood_values, dtype=torch.bool)
        unknown_mask[ood_values > self.tau_high] = True
        tau_high = self.tau_high
        while torch.sum(unknown_mask).item() <= 1:
            tau_high -= self.tau_high
            unknown_mask = ood_values >= tau_high

        both_true = torch.logical_and(known_mask, unknown_mask)
        unknown_mask[both_true] = False

        rejection_mask = (known_mask | unknown_mask)

        return known_mask, unknown_mask, rejection_mask, ood_values


class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, iter_max):
        self.optimizer = optimizer
        self.iter_max = iter_max

        super(CustomLRScheduler, self).__init__(optimizer)

    # update optimizer
    def step(self, iter_num=0, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / self.iter_max) ** (-power)
        # for every parameter in the list
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return self.optimizer


class GaussianMixtureModel():
    def __init__(self, source_class_num, alpha):
        self.alpha = alpha
        self.source_class_num = source_class_num
        self.batch_weight = torch.zeros(source_class_num, dtype=torch.float)
        self.mu = None
        self.C = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def soft_update(self, feat, posterior):
        # Set the desired data type
        dtype = torch.float64  # You can use torch.float128 if supported

        torch.set_printoptions(threshold=float('inf'))

        posterior = posterior.to(dtype)
        feat = feat.to(device=self.device, dtype=dtype)
        self.batch_weight = self.batch_weight.to(device=self.device, dtype=dtype)

        # ---------- Calculate mu ----------
        # Calculate the sum of the posteriors
        batch_weight_new = torch.sum(posterior, dim=0)
        batch_weight_new = batch_weight_new + self.alpha * self.batch_weight

        # Calculate the sum of the weighted features
        weighted_sum = torch.matmul(posterior.T, feat)

        if self.mu != None:
            weighted_sum = self.alpha * self.batch_weight.unsqueeze(1) * self.mu + weighted_sum

        # Calculate mu
        mu_new = weighted_sum / batch_weight_new[:, None]

        # ---------- Calculate the Covariance Matrices ----------
        # Calculate the sum of the outer product
        differences = feat.unsqueeze(1) - mu_new.unsqueeze(0)

        outer_prods = torch.einsum('nmd,nme->nmde', differences, differences)

        epsilon = 1e-6
        eye = torch.eye(differences.shape[2], device=self.device).unsqueeze(0).unsqueeze(0)
        outer_prods = 0.5 * (outer_prods + outer_prods.transpose(-1, -2)) + epsilon * eye

        posterior_expanded = posterior.unsqueeze(-1).unsqueeze(-1)
        weighted_sum = torch.sum(posterior_expanded * outer_prods, dim=0)

        if self.C != None:
            weighted_sum = self.C * self.alpha * self.batch_weight.unsqueeze(1).unsqueeze(2) + weighted_sum

        # Calculate C
        C_new = weighted_sum / batch_weight_new[:, None, None]

        self.batch_weight = batch_weight_new
        self.mu = mu_new
        self.C = C_new

    def get_likelihood(self, feat, mu, C):
        torch.set_printoptions(threshold=float('inf'))
        likelihood = torch.zeros((mu.shape[0], feat.shape[0]))

        # Compute the likelihood of the features for each class
        for i, (mean, cov) in enumerate(zip(mu, C)):
            mean = mean.cpu().detach().numpy() if isinstance(mean, torch.Tensor) else mean
            cov = cov.cpu().detach().numpy() if isinstance(cov, torch.Tensor) else cov
            feat = feat.cpu() if feat.is_cuda else feat
            rv = multivariate_normal(mean, cov, allow_singular=True)
            likelihood[i, :] = torch.from_numpy(rv.logpdf(feat)).type_as(likelihood)

        # for numerical stability
        maximum_likelihood = torch.max(likelihood).item()
        likelihood = likelihood - maximum_likelihood
        likelihood = torch.exp(likelihood)

        # Normalize the likelihood
        likelihood = likelihood / torch.sum(likelihood, axis=0, keepdims=True)

        likelihood = likelihood.T

        return likelihood

    def get_labels(self, feat):
        likelihood = self.get_likelihood(feat, self.mu, self.C)
        max_values, max_indices = torch.max(likelihood, dim=1)

        return max_values, max_indices, likelihood


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)

        if inputs.shape != targets.shape:
            targets = torch.zeros_like(inputs).scatter(1, targets, 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


class HScore(torchmetrics.Metric):
    def __init__(self, known_classes_num, shared_classes_num):
        super(HScore, self).__init__()

        # Number of possible outcomes is total_classes_num
        self.total_classes_num = known_classes_num + 1
        self.shared_classes_num = shared_classes_num

        self.add_state("correct_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")
        self.add_state("total_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        # total_classes_num is the number of Source model outputs including the unknown class.
        for c in range(self.total_classes_num):
            self.total_per_class[c] = self.total_per_class[c] + (target == c).sum()
            self.correct_per_class[c] = self.correct_per_class[c] + ((preds == target) * (target == c)).sum()

    def compute(self):
        # Source-Private classes not included in known_acc
        per_class_acc = self.correct_per_class / (self.total_per_class + 1e-5)
        known_acc = per_class_acc[:self.shared_classes_num].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
        return h_score, known_acc, unknown_acc