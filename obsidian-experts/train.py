"""train.py — trening małego GPT OD ZERA na korpusie PL (L03 + L08 w praktyce).

Pętla (L03): batch → strata cross-entropy → backprop → AdamW → krok. Plus (L08):
warmup + cosine decay LR, early-stopping na val loss (held-out), checkpoint w minimum.
Tokeny z naszego BPE (L05), model z L07 (gpt.py). Domyślnie maleńki — chodzi na CPU w minuty.
To NIE jest GPT-2 124M; to hello-world domykający bramkę „umiem wytrenować własny GPT".

Użycie:
  python train_tokenizer.py            # najpierw tokenizer
  python train.py                      # potem trening (CPU OK)
  python train.py --iters 1000 --n-layer 4 --n-embd 192
"""
import argparse, os, sys, time, math, json
import torch
from gpt import GPT, GPTConfig
from train_tokenizer import encode

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/korpus-pl.txt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print(f"urządzenie: {device}")

    # --- dane: tokenizujemy korpus naszym BPE (L05) ---
    text = open(os.path.join(HERE, args.corpus), encoding="utf-8").read()
    tok = json.load(open(os.path.join(HERE, args.tokenizer), encoding="utf-8"))
    merges, vocab_size = tok["merges"], tok["vocab_size"]
    ids = encode(text, merges)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"vocab={vocab_size} | tokeny: train {len(train_data):,} / val {len(val_data):,}")

    block, batch = args.block, args.batch

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block - 1, (batch,))
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(50)
            for k in range(50):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def lr_at(it):  # warmup + cosine decay (L08)
        warmup = max(1, args.iters // 20)
        if it < warmup:
            return args.lr * it / warmup
        r = (it - warmup) / max(1, args.iters - warmup)
        return args.lr * 0.1 + 0.5 * args.lr * 0.9 * (1 + math.cos(math.pi * r))

    cfg = GPTConfig(vocab_size=vocab_size, block_size=block,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=0.1)
    model = GPT(cfg).to(device)
    print(f"parametry modelu: {model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)

    best_val, t0 = float("inf"), time.time()
    for it in range(args.iters + 1):
        if it % max(1, args.iters // 6) == 0 or it == args.iters:
            L = estimate_loss()
            ppl = math.exp(min(L["val"], 20))
            print(f"  iter {it:4d} | train {L['train']:.3f} | val {L['val']:.3f} | ppl {ppl:.1f} | {time.time()-t0:.0f}s")
            if L["val"] < best_val:                       # early-stopping: zapis w minimum val (L08)
                best_val = L["val"]
                torch.save({"model": model.state_dict(), "config": cfg,
                            "merges": merges, "val_loss": best_val},
                           os.path.join(HERE, args.out))
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\ngotowe. best val loss {best_val:.3f} (ppl {math.exp(min(best_val,20)):.1f}) -> {args.out}")
    print("następne: python sample.py")


if __name__ == "__main__":
    main()
