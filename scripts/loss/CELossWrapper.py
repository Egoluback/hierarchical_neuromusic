import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import CosineSimilarity
from einops import rearrange

class CELossWrapper(CrossEntropyLoss):
    def forward(self, logits, target_ids, **batch) -> Tensor:
        return super().forward(
            logits.permute(0, 2, 1),
            target_ids
        )

class CosineCELossWrapper(CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        return ce_loss + 0.01*cumul_pairwise_loss.mean()