import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

def nce_loss(anchor, positive, negatives) :
    """
    :param anchor: query embed - [batch, embed_size]
    :param positive: [batch, embed_size]
    :param negatives: [batch, k, embed_size]
    """
    pos = torch.sum(anchor * positive, dim=-1).sigmoid().log()
    neg = (- torch.sum(anchor.unsqueeze(dim=-2) * negatives, dim=-1)).sigmoid().log()
    neg = neg.sum(dim=-1)
    return - (pos + neg).mean()


def triplet_loss(anchor, positive, negatives) :
    """
    We found that add all the negative ones together can  yeild relatively better performance.
    :param anchor: user, [batch, embed]
    :param positive: item, [batch, embed]
    :param negatives: item_negs, [batch, num, embed]
    """
    negatives = negatives.permute(1, 0, 2)
    losses = Tensor(0., device=anchor.device)
    for negative in negatives:
        losses += torch.mean(F.triplet_margin_loss(anchor, positive, negative))
    return losses / len(negatives)


def triplet_match_loss(anchor, positive, negatives) :
    """
    :param anchor: user, [batch, embed]
    :param positive: item, [batch, embed]
    :param negatives: item_negs, [batch, num, embed]
    """
    anchor = anchor / anchor.norm(dim=-1, keepdim=True)
    positive = positive / positive.norm(dim=-1, keepdim=True)
    negatives = negatives / negatives.norm(dim=-1, keepdim=True)

    pos_scores = torch.sum(anchor * positive, dim=-1)
    neg_scores = torch.sum(anchor.unsqueeze(1) * negatives, dim=-1)

    pos_targets = torch.ones_like(pos_scores, device=pos_scores.device)
    neg_targets = torch.zeros_like(neg_scores, device=neg_scores.device)

    loss = F.mse_loss(pos_scores, pos_targets) + F.mse_loss(neg_scores, neg_targets)
    return loss


def ncf_bce_loss(positive, negatives) :
    """
    :param positive: scores, [batch, 1]
    :param negatives: scores, [batch, num, 1]
    """
    positive = positive.squeeze(-1)
    negatives = negatives.squeeze(-1)

    pos_targets = torch.ones_like(positive, device=positive.device)
    neg_targets = torch.zeros_like(negatives, device=negatives.device)

    loss = F.binary_cross_entropy_with_logits(positive, pos_targets) +\
           F.binary_cross_entropy_with_logits(negatives, neg_targets)
    return loss


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

    # # nce loss implemented by: https://github.com/sthalles/SimCLR/blob/master/simclr.py
    # # not converge, switched to above impl.
    # def forward(self, batch_sample_one, batch_sample_two):
    #     '''
    #     features shape: n_views*batch_size x feature_dims
    #     examples: [s1-a, s2-a, s3-a, s4-a, s1-b, s2-b, s3-b, s4-b]
    #     '''
    #     features = torch.cat([batch_sample_one, batch_sample_two], dim=0)
    #     labels = torch.cat([torch.arange(features.shape[0])], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.device)

    #     features = F.normalize(features, dim=1)

    #     similarity_matrix = torch.matmul(features, features.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
    #     logits = logits / self.temperature
    #     nce_loss = self.criterion(logits, labels)
    #     return nce_loss


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    code: https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/losses.py
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, batch_sample_one, batch_sample_two):
        z = torch.cat([batch_sample_one, batch_sample_two], dim=0)
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm
        return loss