"""MTA (word / phrase span alignment of hidden states) for causal LMs.

Same loss as `nerkd/mta.py`, adapted to the generative setting where teacher and student see *different*
prompts (the teacher has a few-shot prefix), so the same sentence tokens sit at different positions:
every model gets its own token -> span-id map (B, L, -1 outside spans) built from the same character spans,
span ids are shared. Token weights come from a parameter-free self-attention restricted to the span tokens of
the level; span vector = weighted mean of its tokens; loss = MSE between student / teacher span-span cosine
similarity matrices + 0.1 * (1 - cos(projector(student token), teacher token)).
"""
import torch
import torch.nn.functional as F

from nerkd.mta import compute_token_weights  # parameter-free attention weights, (B, L) from (B, L, H)


def _layout(span_map, mask, K):
    valid = (span_map >= 0) & mask.bool()
    b, t = valid.nonzero(as_tuple=True)
    return b, t, b * K + span_map[b, t]


def _pair_loss(proj, s_h, t_h, sl, tl, s_w, t_w, span_ids_s, span_ids_t, n_spans, span_batch):
    """sl / tl = (b, t) token coordinates of the span tokens in the student / teacher sequences."""
    dev = s_h.device
    s_tok, t_tok = s_h[sl[0], sl[1]].float(), t_h[tl[0], tl[1]].float()
    s_w, t_w = s_w[sl[0], sl[1]], t_w[tl[0], tl[1]]

    def span_mean(tok, w, ids):
        tot = torch.zeros(n_spans, tok.size(-1), device=dev).index_add_(0, ids, tok * w.unsqueeze(-1))
        ws = torch.zeros(n_spans, device=dev).scatter_add_(0, ids, w)
        return tot / ws.clamp(min=1e-5).unsqueeze(-1), ws

    t_span, t_ws = span_mean(t_tok, t_w, span_ids_t)
    s_span, _ = span_mean(s_tok, s_w, span_ids_s)
    s_n, t_n = F.normalize(s_span, dim=-1), F.normalize(t_span, dim=-1)
    pair = (span_batch[:, None] == span_batch[None]) & ~torch.eye(n_spans, dtype=torch.bool, device=dev)
    err = F.mse_loss(s_n @ s_n.T, t_n @ t_n.T, reduction="none")[pair]
    pw = (t_ws[:, None] * t_ws[None])[pair]
    loss = (err * pw).sum() / pw.sum().clamp(min=1e-5)
    if s_tok.size(0) == t_tok.size(0):  # same sentence tokens on both sides -> pair them in order
        tok_l = 1 - F.cosine_similarity(proj(s_tok.to(proj.weight.dtype)).float(), t_tok, dim=-1, eps=1e-5)
        loss = loss + (tok_l * t_w).sum() / t_w.sum().clamp(min=1e-5) / 10.0
    return loss


def _norm_per_sentence(w, coords, B):
    """Weights of the span tokens, normalized to sum to 1 within each sentence."""
    b = coords[0]
    v = w[coords[0], coords[1]]
    s = torch.zeros(B, device=w.device).scatter_add_(0, b, v).clamp(min=1e-5)
    out = torch.zeros_like(w)
    out[coords[0], coords[1]] = v / s[b]
    return out


def level_loss(pairs, projectors, s_map, t_map, s_mask, t_mask, s_hid, t_hid, no_weight=False):
    if not pairs or not (s_map >= 0).any() or not (t_map >= 0).any():
        return s_mask.new_zeros((), dtype=torch.float)
    B = s_map.size(0)
    K = int(max(s_map.max(), t_map.max())) + 1
    sb, st, sg = _layout(s_map, s_mask, K)
    tb, tt, tg = _layout(t_map, t_mask, K)
    uniq = torch.unique(torch.cat([sg, tg]))
    ids_s, ids_t = torch.searchsorted(uniq, sg), torch.searchsorted(uniq, tg)
    span_batch = uniq // K
    s_valid = ((s_map >= 0) & s_mask.bool())
    t_valid = ((t_map >= 0) & t_mask.bool())
    total = 0.0
    for (tl, sl), proj in zip(pairs, projectors):
        if no_weight:
            s_w = _norm_per_sentence(s_valid.float(), (sb, st), B)
            t_w = _norm_per_sentence(t_valid.float(), (tb, tt), B)
        else:  # attention among the span tokens of this level only
            s_w = _norm_per_sentence(compute_token_weights(s_hid[sl].float(), s_valid), (sb, st), B)
            t_w = _norm_per_sentence(compute_token_weights(t_hid[tl].float(), t_valid), (tb, tt), B)
        total = total + _pair_loss(proj, s_hid[sl], t_hid[tl], (sb, st), (tb, tt), s_w, t_w,
                                   ids_s, ids_t, len(uniq), span_batch)
    return total


def mta_loss(projectors, word_pairs, phrase_pairs, b, s_hid, t_hid, no_weight=False):
    """(word + phrase losses) / number of layer pairs; `b` is the collated batch (student + t_* teacher fields)."""
    n = len(word_pairs)
    s_mask = b["attention_mask"]
    t_mask = b.get("t_attention_mask", s_mask)
    word = level_loss(word_pairs, projectors[:n], b["word_span"], b.get("t_word_span", b["word_span"]),
                      s_mask, t_mask, s_hid, t_hid, no_weight)
    phrase = level_loss(phrase_pairs, projectors[n:], b["phrase_span"], b.get("t_phrase_span", b["phrase_span"]),
                        s_mask, t_mask, s_hid, t_hid, no_weight)
    return (word + phrase) / max(n + len(phrase_pairs), 1)
