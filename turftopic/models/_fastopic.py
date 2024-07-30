import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def pairwise_euclidean_distance(x, y):
    cost = (
        torch.sum(x**2, axis=1, keepdim=True)
        + torch.sum(y**2, dim=1)
        - 2 * torch.matmul(x, y.t())
    )
    return cost


class ETP(nn.Module):
    def __init__(
        self,
        sinkhorn_alpha,
        init_a_dist=None,
        init_b_dist=None,
        OT_max_iter=5000,
        stopThr=0.5e-2,
    ):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist
        if init_a_dist is not None:
            self.a_dist = init_a_dist
        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y):
        # Sinkhorn's algorithm
        M = pairwise_euclidean_distance(x, y)
        device = M.device
        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)
        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)
        u = (torch.ones_like(a) / a.size()[0]).to(device)  # Kx1
        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(
                    torch.sum(torch.abs(bb - b), dim=0), p=float("inf")
                )
        transp = u * (K * v.T)
        loss_ETP = torch.sum(transp * M)
        return loss_ETP, transp


class fastopic(nn.Module):
    def __init__(
        self,
        num_topics: int,
        theta_temp: float = 1.0,
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        random_state: Optional[int] = None,
    ):
        super().__init__()

        self.num_topics = num_topics
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.seed = random_state or random.randint(0, 10_000)
        self.epsilon = 1e-12

    def init(self, vocab_size: int, embed_size: int):
        torch.manual_seed(self.seed)
        self.word_embeddings = nn.init.trunc_normal_(
            torch.empty(vocab_size, embed_size)
        )
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(
            F.normalize(self.topic_embeddings)
        )
        self.word_weights = nn.Parameter(
            (torch.ones(vocab_size) / vocab_size).unsqueeze(1)
        )
        self.topic_weights = nn.Parameter(
            (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)
        )
        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

    def get_transp_DT(
        self,
        doc_embeddings,
    ):
        torch.manual_seed(self.seed)
        topic_embeddings = self.topic_embeddings.detach().to(
            doc_embeddings.device
        )
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)
        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        torch.manual_seed(self.seed)
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        # use transport plan as beta
        beta = transp_TW * transp_TW.shape[0]
        return beta

    # only for testing
    def get_theta(self, doc_embeddings, train_doc_embeddings):
        torch.manual_seed(self.seed)
        topic_embeddings = self.topic_embeddings.detach().to(
            doc_embeddings.device
        )
        dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
        train_dist = pairwise_euclidean_distance(
            train_doc_embeddings, topic_embeddings
        )
        exp_dist = torch.exp(-dist / self.theta_temp)
        exp_train_dist = torch.exp(-train_dist / self.theta_temp)
        theta = exp_dist / (exp_train_dist.sum(0))
        theta = theta / theta.sum(1, keepdim=True)
        return theta

    def forward(self, train_bow, doc_embeddings):
        torch.manual_seed(self.seed)
        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(
            self.topic_embeddings, self.word_embeddings
        )
        loss_ETP = loss_DT + loss_TW
        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]
        # Dual Semantic-relation Reconstruction
        recon = torch.matmul(theta, beta)
        loss_DSR = (
            -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()
        )
        loss = loss_DSR + loss_ETP
        rst_dict = {
            "loss": loss,
        }
        return rst_dict
