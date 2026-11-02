"""Logit-level KD divergences for token classification.

DistiLLM divergences are copied from https://github.com/jongwooko/distillm (distillm/losses.py);
`amid` is adapted from https://github.com/voluyen/amid_mta (distillm/losses.py) with its
hyper-parameters passed explicitly instead of through `args`.

All functions take student `logits`, `teacher_logits` of shape (B, L, C) and a dict whose
"label" entry is -100 on positions that must be ignored.
"""
import math

import torch
import torch.nn.functional as F

def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def amid(logits, teacher_logits, no_model_batch, alpha=0.5, lam=0.5, div_name="ab", div_order="pr",
         ab_alpha=0.2, ab_beta=0.7):
    """AMiD (Shin et al., ICLR 2026): divergence between teacher p / student q and the
    alpha-mixture assistant r = mix_alpha(p, q; lam).

    div_name: "fkl" | "ab" (alpha-beta divergence); div_order: "pr" | "qr" | "rp" | "rq".
    """
    p = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    q = F.softmax(logits, dim=-1, dtype=torch.float32)
    logp = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    logq = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)

    # alpha-mixture assistant distribution
    if lam <= 0.0:
        r, logr = q, logq
    elif lam >= 1.0:
        r, logr = p, logp
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
        else:
            t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq
            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
        r = F.softmax(logr_unnorm, dim=-1)
        logr = F.log_softmax(logr_unnorm, dim=-1)

    dists = {"p": (p, logp), "q": (q, logq), "r": (r, logr)}
    if len(div_order) != 2 or any(c not in dists for c in div_order):
        raise ValueError(f"div_order must be one of pr/qr/rp/rq, got {div_order}")
    (a, loga), (_, logb) = dists[div_order[0]], dists[div_order[1]]

    if div_name == "fkl":
        x = torch.masked_fill(a * (loga - logb), inf_mask, 0).sum(-1).view(-1)
        return torch.sum(x * mask.view(-1)) / torch.sum(mask.view(-1))
    if div_name == "ab":
        apb = ab_alpha + ab_beta
        term1 = torch.exp(torch.logsumexp(ab_alpha * loga + ab_beta * logb, dim=-1))
        term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * loga, dim=-1))
        term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logb, dim=-1))
        div = -(term1 - term2 - term3) / (ab_alpha * ab_beta)
        div = torch.where(torch.isfinite(div), div, 0.0)
        mask_sum = mask.sum()
        return (div * mask).sum() / mask_sum if mask_sum > 0 else (div * mask).sum()
    raise ValueError(f"div_name must be fkl or ab, got {div_name}")
