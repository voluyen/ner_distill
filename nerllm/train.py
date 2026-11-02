"""Generative NER on En-ViMedNER: fine-tune a causal LM to map text -> JSON [{"entity", "type"}].

Plain next-token prediction (SFT), loss on the JSON tokens only. Options:
  * --lora-r N          LoRA on a bf16 base; the best adapter is merged into a plain checkpoint
  * --teacher <id>      token-level knowledge distillation on the JSON tokens (teacher and student share the
                        tokenizer, e.g. medgemma-27b-it -> medgemma-4b-it), loss = (1-r)*CE + r*KD.
                        The teacher is used as is (no SFT); --teacher-shots gives it a few-shot prompt
  * --eval-only         skip training, score --model on --test-split (--eval-shots K: few-shot prompt, for the teacher)

Each epoch: dev loss (full dev) + generation F1 on the first --gen-eval-num dev sentences; the epoch with the best
dev F1 is kept and scored on the test split (greedy decoding).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from nerkd import losses as L
from nerllm.data import (CACHE_DIR, SFTDataset, build_prompt, end_ids, load_split, make_collate, pick_shots,
                         render)
from nerllm.metrics import score
from nerllm.mta import mta_loss

KD_LOSSES = {
    "fkl": lambda s, t, b, a: L.forward_kl(s, t, b),
    "rkl": lambda s, t, b, a: L.reverse_kl(s, t, b),
    "skl": lambda s, t, b, a: L.skewed_forward_kl(s, t, b, lam=a.skew_alpha),   # DistiLLM SKL
    "srkl": lambda s, t, b, a: L.skewed_reverse_kl(s, t, b, lam=a.skew_alpha),  # DistiLLM SRKL
    "sym": lambda s, t, b, a: L.symmetric_kl(s, t, b),
    "js": lambda s, t, b, a: L.js_distance(s, t, b),
    "tv": lambda s, t, b, a: L.tv_distance(s, t, b),
    "amid": lambda s, t, b, a: L.amid(s, t, b),  # defaults = the settings used in nerkd (ab, pr, 0.5, 0.5)
    "none": None,  # no logit distillation (MTA only)
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local checkpoint to fine-tune (or to score with --eval-only)")
    p.add_argument("--save", required=True)
    p.add_argument("--data-dir", default=CACHE_DIR)
    p.add_argument("--train-split", default="train")
    p.add_argument("--dev-split", default="dev")
    p.add_argument("--test-split", default="test")
    p.add_argument("--types-in-prompt", action="store_true", help="list the 21 type names in the prompt")
    p.add_argument("--no-chat", action="store_true", help="raw prompt even if the tokenizer has a chat template")
    p.add_argument("--attn", default=None, help="attention implementation, e.g. eager (recommended for Gemma 3)")
    g = p.add_argument_group("distillation")
    g.add_argument("--teacher", default=None, help="HF id / path of the (not fine-tuned) teacher")
    g.add_argument("--teacher-shots", type=int, default=5, help="few-shot examples in the teacher prompt (0 = zero-shot)")
    g.add_argument("--kd-type", default="srkl", choices=list(KD_LOSSES))
    g.add_argument("--w-mta", type=float, default=0.0, help="weight of the MTA span alignment loss (0 = off)")
    g.add_argument("--mta-word-layers", default="31:17", help="teacher:student hidden-state index pairs, word level")
    g.add_argument("--mta-phrase-layers", default="52:28 62:34", help="teacher:student pairs, phrase level")
    g.add_argument("--mta-no-weight", action="store_true", help="ablation: uniform token weights")
    g.add_argument("--kd-ratio", type=float, default=0.5)
    g.add_argument("--skew-alpha", type=float, default=0.1)
    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=2e-5, help="use ~2e-4 with --lora-r")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--warmup-ratio", type=float, default=0.03)
    g.add_argument("--epochs", type=int, default=3)
    g.add_argument("--batch-size", type=int, default=16, help="per step; effective = batch-size * grad-accum")
    g.add_argument("--grad-accum", type=int, default=2)
    g.add_argument("--clip-grad", type=float, default=1.0)
    g.add_argument("--max-length", type=int, default=768)
    g.add_argument("--bf16", action="store_true", help="bf16 autocast (recommended)")
    g.add_argument("--gradient-checkpointing", action="store_true")
    g.add_argument("--lora-r", type=int, default=0, help=">0: LoRA on all linear layers of a bf16 base")
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("evaluation")
    g.add_argument("--eval-only", action="store_true")
    g.add_argument("--eval-batch-size", type=int, default=32)
    g.add_argument("--eval-shots", type=int, default=0, help="few-shot examples in the generation prompt")
    g.add_argument("--gen-max-new-tokens", type=int, default=512)
    g.add_argument("--gen-eval-num", type=int, default=500, help="dev sentences scored by generation each epoch")
    g.add_argument("--train-num", type=int, default=-1, help="subsample (debug)")
    g.add_argument("--dev-num", type=int, default=-1)
    g.add_argument("--test-num", type=int, default=-1)
    a = p.parse_args()
    if a.teacher and a.kd_type == "none" and a.w_mta <= 0:
        p.error("--teacher needs a --kd-type other than none, or --w-mta > 0")
    return a


def load_lm(path, dtype, attn=None):
    """Causal LM for text only. Gemma 3 / MedGemma checkpoints are multimodal (Gemma3ForConditionalGeneration);
    their language model is used through the same class, with text-only inputs."""
    kw = {"torch_dtype": dtype}
    if attn:
        kw["attn_implementation"] = attn
    cfg = AutoConfig.from_pretrained(path)
    if cfg.model_type == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        return Gemma3ForConditionalGeneration.from_pretrained(path, **kw)
    return AutoModelForCausalLM.from_pretrained(path, **kw)


def lora_targets(model):
    """All linear layers; for Gemma 3 only those of the language model (not the vision tower)."""
    if getattr(model.config, "model_type", "") == "gemma3":
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    return "all-linear"


def forward(model, input_ids, attention_mask, hidden=False):
    kw = {}
    if getattr(model.config, "model_type", "") == "gemma3":  # text only: all tokens are "text" tokens
        kw["token_type_ids"] = torch.zeros_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=hidden, **kw)
    return (out.logits, out.hidden_states) if hidden else out.logits


def layer_pairs(spec):
    return [tuple(int(x) for x in p.split(":")) for p in spec.split()]


def n_layers(model):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return cfg.num_hidden_layers


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def shifted(logits, labels):
    """Next-token alignment: position t predicts token t+1. Returns logits at supervised positions and their labels."""
    logits, labels = logits[:, :-1], labels[:, 1:]
    m = labels != -100
    return logits[m], labels[m]


@torch.no_grad()
def dev_loss(model, loader, device, amp):
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if not k.startswith("t_")}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = forward(model, b["input_ids"], b["attention_mask"])
        s, y = shifted(logits, b["labels"])
        tot += F.cross_entropy(s.float(), y, reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def generate(model, tok, rows, args, device, amp):
    """Greedy decoding of the JSON for every row; returns raw strings in the original order."""
    model.eval()
    tok.padding_side = "left"
    shots = pick_shots(load_split(args.train_split, args.data_dir), args.eval_shots) if args.eval_shots else []
    prompts = [render(tok, build_prompt(r["text"], args.types_in_prompt, shots), not args.no_chat) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # similar lengths -> less padding
    outs = [None] * len(rows)
    for i in range(0, len(order), args.eval_batch_size):
        idx = order[i:i + args.eval_batch_size]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            gen = model.generate(**enc, max_new_tokens=args.gen_max_new_tokens, do_sample=False,
                                 eos_token_id=end_ids(tok, not args.no_chat), pad_token_id=tok.pad_token_id, use_cache=True)
        for j, g in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)):
            outs[j] = g
    model.train()
    return outs


def fit(args, tok, dev_rows, device, amp, log):
    """Train, keep the best epoch in args.save (merged into a plain checkpoint when LoRA is used)."""
    train_rows = load_split(args.train_split, args.data_dir, args.train_num)
    shots = pick_shots(train_rows, args.teacher_shots) if args.teacher else []
    use_mta = args.w_mta > 0
    mk = lambda rows, sp=False: SFTDataset(rows, tok, args.max_length, args.types_in_prompt, not args.no_chat, shots,
                                           with_spans=sp, cache_dir=args.data_dir)
    train_set, dev_set = mk(train_rows, use_mta), mk(dev_rows)
    log(f"train {len(train_set)} (dropped {train_set.dropped} > {args.max_length} tokens) | "
        f"dev {len(dev_set)}")
    ex = train_set[0]
    log("example input:\n" + tok.decode(ex["input_ids"]))
    if "t_input_ids" in ex:
        log("example teacher input:\n" + tok.decode(ex["t_input_ids"]))
    collate = make_collate(tok.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, collate_fn=collate)

    lora = args.lora_r > 0
    model = load_lm(args.model, torch.bfloat16 if lora else torch.float32, args.attn)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if lora:
            model.enable_input_require_grads()
    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=lora_targets(model)))
        log(f"LoRA r={args.lora_r}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M trainable")
    model.to(device).train()

    teacher, V = None, None
    if args.teacher:
        t_tok = AutoTokenizer.from_pretrained(args.teacher)
        assert t_tok.get_vocab() == tok.get_vocab(), "teacher and student must share the tokenizer"
        teacher = load_lm(args.teacher, torch.bfloat16, args.attn).to(device)
        teacher.eval().requires_grad_(False)
        # checkpoints of one family may pad the output layer differently: compare on the common vocabulary
        V = min(model.get_output_embeddings().weight.size(0), teacher.get_output_embeddings().weight.size(0))
        kd_fn = KD_LOSSES[args.kd_type]

    projectors, word_pairs, phrase_pairs = None, [], []
    if use_mta:
        assert teacher is not None, "--w-mta needs --teacher"
        word_pairs, phrase_pairs = layer_pairs(args.mta_word_layers), layer_pairs(args.mta_phrase_layers)
        assert word_pairs or phrase_pairs, "MTA needs at least one layer pair"
        n_t, n_s = n_layers(teacher), n_layers(model)
        for t_l, s_l in word_pairs + phrase_pairs:
            assert 0 <= t_l <= n_t and 0 <= s_l <= n_s, f"layer pair {t_l}:{s_l} out of range (teacher {n_t}, student {n_s})"
        h_s, h_t = model.get_output_embeddings().in_features, teacher.get_output_embeddings().in_features
        projectors = torch.nn.ModuleList(torch.nn.Linear(h_s, h_t) for _ in word_pairs + phrase_pairs).to(device)
        log(f"MTA word pairs {word_pairs} | phrase pairs {phrase_pairs} | weight {args.w_mta}")

    params = [p for p in model.parameters() if p.requires_grad] + (list(projectors.parameters()) if use_mta else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total), total)

    best_f1, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for it, b in enumerate(train_loader, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = forward(model, b["input_ids"], b["attention_mask"], hidden=use_mta)
                logits, s_hid = out if use_mta else (out, None)
                s, y = shifted(logits, b["labels"])
                ce = F.cross_entropy(s.float(), y)
                loss, kd, mta = ce, torch.zeros((), device=device), torch.zeros((), device=device)
                if teacher is not None:
                    with torch.no_grad():
                        # the teacher prompt may be longer (few-shot) but the supervised tokens are the same
                        out = forward(teacher, b.get("t_input_ids", b["input_ids"]),
                                      b.get("t_attention_mask", b["attention_mask"]), hidden=use_mta)
                        t_logits, t_hid = out if use_mta else (out, None)
                        t = shifted(t_logits, b.get("t_labels", b["labels"]))[0]
                    # (N, V) -> (1, N, V): the KD losses take (B, L, C) logits and a dict with "label"
                    if kd_fn is not None:
                        kd = kd_fn(s[:, :V].float()[None], t[:, :V].float()[None], {"label": y[None]}, args)
                    if use_mta:
                        mta = mta_loss(projectors, word_pairs, phrase_pairs, b, s_hid, t_hid, args.mta_no_weight)
                    loss = (1 - args.kd_ratio) * ce + args.kd_ratio * (kd + args.w_mta * mta)
            (loss / args.grad_accum).backward()
            if it % args.grad_accum == 0 or it == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log(f"epoch {epoch} step {step}/{total} | loss {loss.item():.4f} | ce {ce.item():.4f} "
                        f"| kd {kd.item():.4f} | mta {mta.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        dl = dev_loss(model, dev_loader, device, amp)
        res = score(generate(model, tok, dev_rows[:args.gen_eval_num], args, device, amp), dev_rows[:args.gen_eval_num])
        log(f"== epoch {epoch} ({time.time() - t0:.0f}s) | dev loss {dl:.4f} | dev(gen, {args.gen_eval_num}) {res}")
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            (model.save_pretrained(os.path.join(args.save, "adapter")) if lora else model.save_pretrained(args.save))
            tok.save_pretrained(args.save)
            log(f"   saved best (dev f1 {best_f1})")

    del model, optimizer, teacher
    torch.cuda.empty_cache()
    if lora:  # merge the best adapter into the base -> a plain checkpoint, usable as a KD teacher
        from peft import PeftModel
        base = load_lm(args.model, torch.bfloat16, args.attn)
        merged = PeftModel.from_pretrained(base, os.path.join(args.save, "adapter")).merge_and_unload()
        merged.save_pretrained(args.save)
        del merged, base


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.bf16 and device.type == "cuda"
    log_f = open(os.path.join(args.save, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(json.dumps(vars(args)))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev_rows = load_split(args.dev_split, args.data_dir, args.dev_num)
    test_rows = load_split(args.test_split, args.data_dir, args.test_num)
    log(f"dev {len(dev_rows)} | test {len(test_rows)} sentences")

    if not args.eval_only:
        fit(args, tok, dev_rows, device, amp, log)
        torch.cuda.empty_cache()
        ckpt = args.save
    else:
        ckpt = args.model

    # final scoring with the selected checkpoint
    model = load_lm(ckpt, torch.bfloat16 if amp else torch.float32, args.attn).to(device)
    dev_res = score(generate(model, tok, dev_rows, args, device, amp), dev_rows)
    preds = generate(model, tok, test_rows, args, device, amp)
    test_res = score(preds, test_rows)
    log(f"BEST | dev {dev_res} | test {test_res}")
    with open(os.path.join(args.save, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r, p in zip(test_rows, preds):
            f.write(json.dumps({"id": r["id"], "text": r["text"], "gold": [{"entity": e["entity"], "type": e["type"]}
                                for e in r["entities"]], "pred": p}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.save, "results.json"), "w") as f:
        json.dump({"dev": dev_res, "test": test_res,
                   "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}, f, indent=2)


if __name__ == "__main__":
    main()
