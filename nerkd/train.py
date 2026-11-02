"""Encoder -> encoder knowledge distillation for NER (token classification).

  * no --teacher         -> plain fine-tuning (teacher SFT / student SFT baseline)
  * --teacher --kd-type  -> loss = (1 - kd_ratio) * CE + kd_ratio * (T^2 * KD + w_mta * MTA)
      KD : logit divergence on labelled tokens (first sub-token of each word)
           DistiLLM: fkl | rkl | skl | srkl | sym | js | tv, AMiD: amid
      MTA: word/phrase span alignment of hidden states (enabled when --w-mta > 0)
Teacher and student must share the tokenizer. Checkpoint chosen by dev F1, reported on test.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from nerkd.data import NERDataset, load_conll, make_collate
from nerkd.losses import (amid, forward_kl, js_distance, reverse_kl, skewed_forward_kl,
                          skewed_reverse_kl, symmetric_kl, tv_distance)
from nerkd.mta import mta_loss
from nerkd.ood import OOD, evaluate_ood

KD_LOSSES = {
    "none": None,
    "fkl": lambda s, t, b, a: forward_kl(s, t, b),
    "rkl": lambda s, t, b, a: reverse_kl(s, t, b),
    "skl": lambda s, t, b, a: skewed_forward_kl(s, t, b, lam=a.skew_alpha),   # DistiLLM SKL
    "srkl": lambda s, t, b, a: skewed_reverse_kl(s, t, b, lam=a.skew_alpha),  # DistiLLM SRKL
    "sym": lambda s, t, b, a: symmetric_kl(s, t, b),
    "js": lambda s, t, b, a: js_distance(s, t, b),
    "tv": lambda s, t, b, a: tv_distance(s, t, b),
    "amid": lambda s, t, b, a: amid(s, t, b, alpha=a.amid_alpha, lam=a.amid_lam, div_name=a.amid_div,
                                     div_order=a.amid_order, ab_alpha=a.ab_alpha, ab_beta=a.ab_beta),
}


def layer_pairs(spec):
    """'6:3 10:5' -> [(6, 3), (10, 5)]  (teacher_layer:student_layer, hidden_states index)."""
    return [tuple(int(x) for x in p.split(":")) for p in spec.split()] if spec else []


def get_args():
    p = argparse.ArgumentParser()
    g = p.add_argument_group("model / data")
    g.add_argument("--model", required=True, help="student (or model to fine-tune)")
    g.add_argument("--teacher", default=None, help="fine-tuned teacher checkpoint; enables KD")
    g.add_argument("--dataset", default="eriktks/conll2003")
    g.add_argument("--save", required=True)
    g.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "..", "cache"))
    g = p.add_argument_group("KD")
    g.add_argument("--kd-type", default="none", choices=list(KD_LOSSES))
    g.add_argument("--kd-ratio", type=float, default=0.5, help="weight of distillation vs. CE on gold labels")
    g.add_argument("--kd-temperature", type=float, default=1.0)
    g.add_argument("--skew-alpha", type=float, default=0.1, help="DistiLLM skew KL alpha")
    g = p.add_argument_group("AMiD")
    g.add_argument("--amid-alpha", type=float, default=0.5)
    g.add_argument("--amid-lam", type=float, default=0.5)
    g.add_argument("--amid-div", default="ab", choices=["ab", "fkl"])
    g.add_argument("--amid-order", default="pr", choices=["pr", "qr", "rp", "rq"])
    g.add_argument("--ab-alpha", type=float, default=0.2)
    g.add_argument("--ab-beta", type=float, default=0.7)
    g = p.add_argument_group("MTA")
    g.add_argument("--w-mta", type=float, default=0.0, help="weight of the span alignment loss (0 = off)")
    g.add_argument("--mta-word-layers", default="6:3", help="teacher:student hidden-state pairs, word level")
    g.add_argument("--mta-phrase-layers", default="10:5 12:6", help="teacher:student pairs, phrase level")
    g.add_argument("--mta-no-weight", action="store_true", help="ablation: uniform token weights")
    g.add_argument("--projector-lr", type=float, default=5e-4)
    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup-ratio", type=float, default=0.1)
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--eval-batch-size", type=int, default=128)
    g.add_argument("--max-length", type=int, default=256)
    g.add_argument("--label-pos", default="first", choices=["first", "last"],
                   help="sub-token carrying each word's label; 'last' for causal decoders (Qwen)")
    g.add_argument("--teacher-bf16", action="store_true", help="load the frozen teacher in bf16 (7B teachers)")
    g.add_argument("--lora-r", type=int, default=0, help=">0: LoRA fine-tuning of a bf16 base (7B teacher SFT)")
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing of the trained model")
    g.add_argument("--clip-grad", type=float, default=1.0)
    g.add_argument("--bf16", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--train-num", type=int, default=-1, help="subsample train set (debug)")
    g.add_argument("--eval-num", type=int, default=-1, help="subsample dev/test (debug)")
    g.add_argument("--ood-eval", nargs="*", default=[], choices=list(OOD),
                   help="also evaluate the best checkpoint OOD (IWD protocol), e.g. --ood-eval wnut17 ontonotes")
    args = p.parse_args()
    if args.teacher is None and (args.kd_type != "none" or args.w_mta > 0):
        p.error("--kd-type / --w-mta need --teacher")
    if args.teacher is not None and args.kd_type == "none" and args.w_mta <= 0:
        p.error("--teacher given but neither --kd-type nor --w-mta is set")
    if args.lora_r > 0 and args.teacher is not None:
        p.error("--lora-r is for teacher fine-tuning only")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, id2label, device, amp_dtype):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        pred = logits.argmax(-1).cpu().tolist()
        for p_row, g_row in zip(pred, batch["labels"].cpu().tolist()):
            keep = [(id2label[p], id2label[g]) for p, g in zip(p_row, g_row) if g != -100]
            preds.append([p for p, _ in keep])
            golds.append([g for _, g in keep])
    model.train()
    return {
        "precision": round(100 * precision_score(golds, preds), 2),
        "recall": round(100 * recall_score(golds, preds), 2),
        "f1": round(100 * f1_score(golds, preds), 2),
    }, classification_report(golds, preds, digits=4)


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else None
    log_f = open(os.path.join(args.save, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(json.dumps(vars(args)))
    ds, labels = load_conll(args.dataset)
    id2label = dict(enumerate(labels))
    label2id = {l: i for i, l in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def load_base():
        return AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(labels), id2label=id2label, label2id=label2id,
            torch_dtype=torch.bfloat16 if args.lora_r > 0 else None)

    student = load_base()
    student.config.nerkd_label_pos = args.label_pos  # read back by `python -m nerkd.ood`
    if args.grad_ckpt:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora_r > 0:
        from peft import LoraConfig, TaskType, get_peft_model
        student = get_peft_model(student, LoraConfig(
            task_type=TaskType.TOKEN_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p_ in student.parameters():  # trainable LoRA + head in fp32, frozen base stays bf16
            if p_.requires_grad:
                p_.data = p_.data.float()
        n_train = sum(p_.numel() for p_ in student.parameters() if p_.requires_grad)
        log(f"LoRA r={args.lora_r}: {n_train / 1e6:.1f}M trainable parameters")
    student.to(device)

    teacher = None
    if args.teacher:
        # teacher and student must share the tokenizer: same input_ids feed both models
        t_tok = AutoTokenizer.from_pretrained(args.teacher)
        assert t_tok.get_vocab() == tokenizer.get_vocab(), \
            f"teacher tokenizer ({args.teacher}) differs from student ({args.model})"
        teacher = AutoModelForTokenClassification.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16 if args.teacher_bf16 else None).to(device)
        teacher.eval().requires_grad_(False)
        assert teacher.config.id2label == student.config.id2label, "teacher/student label maps differ"

    use_mta = args.w_mta > 0
    word_pairs, phrase_pairs = layer_pairs(args.mta_word_layers), layer_pairs(args.mta_phrase_layers)
    projectors = None
    if use_mta:
        n_t, n_s = teacher.config.num_hidden_layers, student.config.num_hidden_layers
        for t, s in word_pairs + phrase_pairs:
            assert 0 <= t <= n_t and 0 <= s <= n_s, f"layer pair {t}:{s} out of range (teacher {n_t}, student {n_s})"
        assert word_pairs or phrase_pairs, "MTA needs at least one layer pair"
        projectors = torch.nn.ModuleList(
            torch.nn.Linear(student.config.hidden_size, teacher.config.hidden_size)
            for _ in word_pairs + phrase_pairs).to(device)
        log(f"MTA word pairs {word_pairs} | phrase pairs {phrase_pairs} | weight {args.w_mta}")

    def subset(split, n):
        return split if n < 0 else split.select(range(min(n, len(split))))

    train_set = NERDataset(subset(ds["train"], args.train_num), tokenizer, args.max_length,
                           with_spans=use_mta, cache_dir=args.cache_dir, label_pos=args.label_pos)
    dev_set = NERDataset(subset(ds["validation"], args.eval_num), tokenizer, args.max_length, label_pos=args.label_pos)
    test_set = NERDataset(subset(ds["test"], args.eval_num), tokenizer, args.max_length, label_pos=args.label_pos)
    log(f"train {len(train_set)} | dev {len(dev_set)} | test {len(test_set)} | labels {labels}")

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    if projectors is not None:
        params.append({"params": list(projectors.parameters()), "lr": args.projector_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    kd_fn = KD_LOSSES[args.kd_type]
    T = args.kd_temperature
    best_f1, step = -1.0, 0
    student.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                s_out = student(**inputs, output_hidden_states=use_mta)
                s_logits = s_out.logits.float()
                ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), batch["labels"].view(-1), ignore_index=-100)
                loss = ce
                kd = mta = torch.zeros((), device=device)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(**inputs, output_hidden_states=use_mta)
                    if kd_fn is not None:
                        kd = kd_fn(s_logits / T, t_out.logits.float() / T, {"label": batch["labels"]}, args) * T * T
                    if use_mta:
                        mta = mta_loss(projectors, word_pairs, phrase_pairs, batch, s_out.hidden_states,
                                       t_out.hidden_states, no_span_weight=args.mta_no_weight)
                    loss = (1 - args.kd_ratio) * ce + args.kd_ratio * (kd + args.w_mta * mta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log(f"epoch {epoch} step {step}/{total_steps} | loss {loss.item():.4f} | ce {ce.item():.4f} "
                    f"| kd {kd.item():.4f} | mta {mta.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        dev_res, _ = evaluate(student, dev_loader, id2label, device, amp_dtype)
        log(f"== epoch {epoch} ({time.time() - t0:.0f}s) | dev {dev_res}")
        if dev_res["f1"] > best_f1:
            best_f1 = dev_res["f1"]
            student.save_pretrained(os.path.join(args.save, "adapter") if args.lora_r > 0 else args.save)
            tokenizer.save_pretrained(args.save)
            log(f"   saved best (dev f1 {best_f1})")

    # final test with the checkpoint selected on dev F1
    if args.lora_r > 0:  # merge the best adapter into the base -> plain checkpoint usable as a KD teacher
        from peft import PeftModel
        adapter = os.path.join(args.save, "adapter")
        merged = PeftModel.from_pretrained(load_base(), adapter).merge_and_unload()
        merged.config.nerkd_label_pos = args.label_pos
        merged.save_pretrained(args.save)
        del merged
    best = AutoModelForTokenClassification.from_pretrained(
        args.save, torch_dtype=torch.bfloat16 if args.lora_r > 0 else None).to(device)
    dev_res, _ = evaluate(best, dev_loader, id2label, device, amp_dtype)
    test_res, report = evaluate(best, test_loader, id2label, device, amp_dtype)
    log(f"BEST | dev {dev_res} | test {test_res}\n{report}")
    res = {"dev": dev_res, "test": test_res, "params_M": round(sum(p.numel() for p in best.parameters()) / 1e6, 1)}
    if args.ood_eval:
        res["ood"] = evaluate_ood(best, tokenizer, id2label, collate, device, amp_dtype, names=args.ood_eval,
                                  max_length=args.max_length, batch_size=args.eval_batch_size,
                                  label_pos=args.label_pos)
        log(f"OOD | {res['ood']}")
    with open(os.path.join(args.save, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
