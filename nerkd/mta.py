"""MTA: multi-level (word / phrase) span alignment of hidden states, adapted for encoders.

Adapted from https://github.com/voluyen/amid_mta (span_utils.py). The original recovers spans
from character offsets of decoded text on every step; for NER the input is already split into
words, so spans are precomputed once per sentence (see `nerkd.data.compute_spans`) and passed in as
token -> span-id maps of shape (B, L) with -1 on tokens outside any span.

For each (teacher layer, student layer) pair:
  * token importance weights from a parameter-free self-attention over the layer's hidden states
    (or uniform weights with `no_span_weight`),
  * span representation = importance-weighted mean of its tokens,
  * loss = MSE between the student and teacher span-span cosine-similarity matrices (within each
    sentence, weighted by span importance) + 0.1 * (1 - cos(projector(student token), teacher token)).
"""
import torch
import torch.nn.functional as F


def compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    q = hidden_state / std
    scores = torch.matmul(q, q.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)
    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    diag = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag.unsqueeze(0), float("-inf"))
    attn = torch.nan_to_num(F.softmax(scores, dim=-1)) * mask
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return attn.mean(dim=1).detach()  # (B, L)


def _span_layout(span_map, attention_mask):
    """Flatten the tokens that belong to a span; give every (sentence, span) a global id."""
    B, L = span_map.shape
    valid = (span_map >= 0) & attention_mask.bool()
    b_idx, t_idx = valid.nonzero(as_tuple=True)
    flat_idx = b_idx * L + t_idx
    global_span = b_idx * (int(span_map.max()) + 1) + span_map[b_idx, t_idx]
    _, span_ids = torch.unique(global_span, return_inverse=True)
    n_spans = int(span_ids.max()) + 1
    span_batch = torch.empty(n_spans, dtype=torch.long, device=span_map.device).scatter_(0, span_ids, b_idx)
    return flat_idx, b_idx, span_ids, n_spans, span_batch


def _normalize_per_sentence(weights, flat_idx, b_idx, B):
    """Token weights restricted to span tokens, normalized to sum to 1 within each sentence."""
    w = weights.reshape(-1)[flat_idx].float()
    sums = torch.zeros(B, device=w.device).scatter_add_(0, b_idx, w).clamp(min=1e-5)
    return w / sums[b_idx]


def _pair_loss(projector, s_hidden, t_hidden, flat_idx, s_w, t_w, span_ids, n_spans, span_batch):
    t_tok = t_hidden.flatten(0, 1)[flat_idx].float()
    s_tok = s_hidden.flatten(0, 1)[flat_idx].float()
    device = t_tok.device

    def span_mean(tok, w):
        total = torch.zeros(n_spans, tok.size(-1), device=device).index_add_(0, span_ids, tok * w.unsqueeze(-1))
        w_sum = torch.zeros(n_spans, device=device).scatter_add_(0, span_ids, w)
        return total / w_sum.clamp(min=1e-5).unsqueeze(-1), w_sum

    t_span, t_wsum = span_mean(t_tok, t_w)
    s_span, _ = span_mean(s_tok, s_w)
    s_n, t_n = F.normalize(s_span, dim=-1), F.normalize(t_span, dim=-1)
    pair_mask = (span_batch.unsqueeze(1) == span_batch.unsqueeze(0)) & \
        ~torch.eye(n_spans, dtype=torch.bool, device=device)
    sim_err = F.mse_loss(s_n @ s_n.T, t_n @ t_n.T, reduction="none")[pair_mask]
    pair_w = (t_wsum.unsqueeze(1) * t_wsum.unsqueeze(0))[pair_mask]
    span_loss = (sim_err * pair_w).sum() / pair_w.sum().clamp(min=1e-5)

    token_loss = 1 - F.cosine_similarity(projector(s_tok.to(projector.weight.dtype)).float(), t_tok, dim=-1, eps=1e-5)
    token_loss = (token_loss * t_w).sum() / t_w.sum().clamp(min=1e-5)
    return span_loss + token_loss / 10.0


def level_loss(pairs, projectors, span_map, attention_mask, s_hidden_states, t_hidden_states, no_span_weight=False):
    """Sum of the alignment losses over (teacher_layer, student_layer) pairs for one span level."""
    if not pairs or not ((span_map >= 0) & attention_mask.bool()).any():
        return attention_mask.new_zeros((), dtype=torch.float)
    flat_idx, b_idx, span_ids, n_spans, span_batch = _span_layout(span_map, attention_mask)
    B = attention_mask.size(0)
    total = 0.0
    for (t_layer, s_layer), proj in zip(pairs, projectors):
        if no_span_weight:
            t_w = s_w = _normalize_per_sentence(attention_mask.float(), flat_idx, b_idx, B)
        else:
            t_w = _normalize_per_sentence(compute_token_weights(t_hidden_states[t_layer].float(), attention_mask),
                                          flat_idx, b_idx, B)
            s_w = _normalize_per_sentence(compute_token_weights(s_hidden_states[s_layer].float(), attention_mask),
                                          flat_idx, b_idx, B)
        total = total + _pair_loss(proj, s_hidden_states[s_layer], t_hidden_states[t_layer], flat_idx,
                                   s_w, t_w, span_ids, n_spans, span_batch)
    return total


def mta_loss(projectors, word_pairs, phrase_pairs, batch, s_hidden_states, t_hidden_states, no_span_weight=False):
    """(word-level + phrase-level loss) / number of layer pairs, as in the original MTA."""
    n_word = len(word_pairs)
    word = level_loss(word_pairs, projectors[:n_word], batch["word_span"], batch["attention_mask"],
                      s_hidden_states, t_hidden_states, no_span_weight)
    phrase = level_loss(phrase_pairs, projectors[n_word:], batch["phrase_span"], batch["attention_mask"],
                        s_hidden_states, t_hidden_states, no_span_weight)
    return (word + phrase) / max(n_word + len(phrase_pairs), 1)
