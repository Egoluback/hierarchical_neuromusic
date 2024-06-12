import numpy as np
import torch
from einops import rearrange
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from torch.nn import CosineSimilarity, CrossEntropyLoss


class CELossWrapper(CrossEntropyLoss):
    def forward(self, logits, target_ids, **batch) -> Tensor:
        return super().forward(
            logits.permute(0, 2, 1),
            target_ids
        )


class CosineCELossWrapper(CrossEntropyLoss):
    def __init__(self, lr=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.cosine = CosineSimilarity()

    def calc_pairwise_cosine(self, logits_hidden) -> Tensor:
        logits_hidden = rearrange(logits_hidden, '(n s) b d -> (n b) s d', s=2)

        return self.cosine(logits_hidden[:, 0], logits_hidden[:, 1])

    def forward(self, logits, target_ids, logits_hidden, **batch) -> Tensor:
        ce_loss = super().forward(
            logits.permute(0, 2, 1),
            target_ids
        )

        cumul_pairwise_loss = torch.cat([self.calc_pairwise_cosine(logit_hidden) for logit_hidden in logits_hidden])

        return ce_loss + self.lr * cumul_pairwise_loss.mean()


class ExpCosineCELossWrapper(CrossEntropyLoss):
    def __init__(self, lr=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.lr = lr

    def calc_pairwise_cosine(self, logits_hidden) -> Tensor:
        eye_mask = np.where(~np.eye(logits_hidden.shape[0], dtype=bool))
        logits_hidden = logits_hidden.cpu().detach()
        logits_cosine = torch.stack(
            [torch.Tensor(-cosine_similarity(rearrange(logits_hidden, 'n b d -> b n d')[b])[
                eye_mask]) for b in range(logits_hidden.shape[1])])

        logits_cosine = rearrange(logits_cosine, 'b (a n) -> b a n', a=logits_hidden.shape[0])

        return torch.exp(logits_cosine).sum(axis=1).sum(axis=1)

    def forward(self, logits, target_ids, logits_hidden, **batch) -> Tensor:
        ce_loss = super().forward(
            logits.permute(0, 2, 1),
            target_ids
        )

        cumul_pairwise_loss = torch.cat([self.calc_pairwise_cosine(logit_hidden) for logit_hidden in logits_hidden])

        return ce_loss + self.lr * cumul_pairwise_loss.mean()
