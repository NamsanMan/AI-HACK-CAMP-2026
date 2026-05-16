import torch


def binary_accuracy(logits_or_probs, targets, threshold: float = 0.5):
    probs = torch.sigmoid(logits_or_probs) if logits_or_probs.min() < 0 or logits_or_probs.max() > 1 else logits_or_probs
    return ((probs >= threshold).float() == targets.float()).float().mean()


def negative_pearson_loss(pred, target, eps: float = 1e-8):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    corr = (pred * target).sum(dim=-1) / (pred.norm(dim=-1) * target.norm(dim=-1) + eps)
    return 1.0 - corr.mean()

