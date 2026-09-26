# Autor: Adam Skrodzki
"""Judge v1: klasyfikator meter/mode/type na trunku GPT (verbatim core/gpt.py).
Ciało melodii (nagłówki usunięte) -> znaki -> GPT -> pooling (mean|attn) -> 3 liniowe głowy.
Split po tune_id (settingi tej samej melodii nie mogą przeciekać do val).
Czysty PyTorch. Opcjonalnie --init-from: trunk startuje z wag pretrenowanego eksperta.
Użycie: python src/tools/train_judge.py [--csv data/tunes.csv] [--out data/models/judge_v2.pt]
        [--init-from data/models/jig_ckpt.pt] [--seed 42]
"""
import argparse, collections, csv, os, random, sys, time
from contextlib import nullcontext
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv.field_size_limit(10**7)
import torch
import torch.nn.functional as F
from core.gpt import GPTConfig
from core.judge import JudgeGPT, HEADS
from core.abc_corpus import parse_tune_row

MIN_BODY, MAX_BODY = 40, 700   # jak prepare_data.py
BLOCK = 256
DEVICE = "cpu"                 # ustawiane w main(): "cuda" gdy dostępne (autodetekcja jak train_gpt.py)

def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            body, meter, mode, typ = parse_tune_row(row)
            if MIN_BODY <= len(body) <= MAX_BODY:
                rows.append({"body": body, "tune_id": row["tune_id"],
                             "meter": meter, "mode": mode, "type": typ})
    return rows

def group_split(rows, seed):
    """90/10 po unikalnych tune_id (deterministycznie)."""
    ids = sorted({r["tune_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, len(ids) // 10)
    val_ids = set(ids[:n_val])
    tr = [i for i, r in enumerate(rows) if r["tune_id"] not in val_ids]
    va = [i for i, r in enumerate(rows) if r["tune_id"] in val_ids]
    return tr, va

def encode_train(rows, idxs, stoi, block):
    """Trening: losowe okno block znaków z każdego ciała (augmentacja — przez epoki
    model widzi całe ciało, nie tylko pierwsze block znaków). Krótsze ciała = 1 okno.
    Padding na końcu; 0 w idx = padding."""
    B = len(idxs)
    idx = torch.zeros(B, block, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(B, block, device=DEVICE)
    for b, i in enumerate(idxs):
        body = rows[i]["body"]
        start = random.randint(0, max(0, len(body) - block))
        ids = [stoi[c] for c in body[start:start + block] if c in stoi]
        if not ids:
            ids = [stoi[" "]]
        idx[b, :len(ids)] = torch.tensor(ids, device=DEVICE)
        mask[b, :len(ids)] = 1.0
    return idx, mask

def windows(body, block):
    """Wszystkie okna niepokrywające (ostatnie może być krótsze)."""
    return [body[s:s + block] for s in range(0, len(body), block)] or [" "]

def evaluate(model, rows, idxs, classes, stoi, bs=64, block=BLOCK):
    """Eval: KAŻDE ciało dzielone na wszystkie okna (pełne pokrycie, deterministycznie),
    softmax uśredniany po oknach -> jedna predykcja na ciało."""
    model.eval()
    samples = [(i, w) for i in idxs for w in windows(rows[i]["body"], block)]
    agg = {i: {h: torch.zeros(len(classes[h])) for h in HEADS} for i in idxs}
    cnt = collections.Counter()
    with torch.no_grad():
        for s in range(0, len(samples), bs):
            chunk = samples[s:s+bs]
            idx = torch.zeros(len(chunk), block, dtype=torch.long, device=DEVICE)
            mask = torch.zeros(len(chunk), block, device=DEVICE)
            for b, (i, w) in enumerate(chunk):
                ids = [stoi[c] for c in w if c in stoi] or [stoi[" "]]
                idx[b, :len(ids)] = torch.tensor(ids, device=DEVICE)
                mask[b, :len(ids)] = 1.0
            logits = model(idx, mask)
            for b, (i, _) in enumerate(chunk):
                for h in HEADS:
                    agg[i][h] += F.softmax(logits[h][b], dim=-1).cpu()
                cnt[i] += 1
    correct = collections.Counter(); n = 0
    confusion = {h: collections.Counter() for h in HEADS}
    for i in idxs:
        n += 1
        for h in HEADS:
            cls = classes[h]
            mean_p = agg[i][h] / cnt[i]
            pred = cls[int(mean_p.argmax())]
            true = rows[i][h]
            correct[h] += pred == true
            if pred != true:
                confusion[h].update([(true, pred)])
    model.train()
    return {h: correct[h] / n for h in HEADS}, confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/tunes.csv")
    ap.add_argument("--out", default="data/models/judge_v2.pt")
    ap.add_argument("--init-from", default=None, help="ckpt eksperta: pretrenowany trunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block", type=int, default=256, help="długość okna ciała (znaki)")
    ap.add_argument("--pool", choices=["mean", "attn"], default="mean",
                    help="pooling po pozycjach: mean (domyślny w dotychczasowych sędziach) albo attention-pool")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--train-eval-n", type=int, default=5000,
                    help="ile przykładów train do logowania acc (0 = cały train)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    torch.manual_seed(a.seed)
    print(f"urządzenie: {DEVICE} | bf16: {use_bf16}")

    print(f"czytam {a.csv} ...")
    rows = load_rows(a.csv)
    tr, va = group_split(rows, a.seed)
    steps_per_epoch = len(tr) // a.batch
    TRAIN_EVAL = tr if a.train_eval_n <= 0 else tr[:a.train_eval_n]  # stała próbka train do logu
    print(f"melodii (próbek): {len(rows)} łącznie | train {len(tr)} | val {len(va)} (split po tune_id)")
    print(f"batch: {a.batch} próbek/krok | okno: {a.block} znaków | lr: {a.lr}")
    print(f"próbkowanie losowe: pełne 'przejście' po train ≈ co {steps_per_epoch} kroków "
          f"({len(tr)}/{a.batch}); ten sam przykład wraca średnio co ~{steps_per_epoch} kroków, "
          f"ale okno (fragment ciała) losowane jest za każdym razem na nowo")
    print(f"logowanie acc co {a.eval_every} kroków: train = stała próbka {len(TRAIN_EVAL)} "
          f"przykładów, val = wszystkie {len(va)}")

    chars = sorted({c for i in tr for c in rows[i]["body"]})
    stoi = {c: j + 1 for j, c in enumerate(chars)}   # 0 = padding
    print(f"słownik ciał: {len(chars)} znaków (+padding)")

    classes = {h: sorted({r[h] for r in rows}) for h in HEADS}
    n_classes = {h: len(c) for h, c in classes.items()}
    print("klasy: " + " | ".join(f"{h}: {n_classes[h]}" for h in HEADS))

    cfg = GPTConfig(vocab_size=len(chars) + 1, block_size=a.block)
    model = JudgeGPT(cfg, n_classes, pool=a.pool).to(DEVICE)

    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        src = ck["model"]
        dst = model.state_dict()
        copied = [k for k in dst if k.startswith("gpt.") and k != "gpt.head.weight"
                  and k in src and src[k].shape == dst[k].shape]
        for k in copied:
            dst[k] = src[k]
        model.load_state_dict(dst)
        print(f"trunk zainicjowany z {a.init_from} ({len(copied)} tensorów, LM-head pominięty)")
    print(f"parametry: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.1)

    def train_step():
        chunk = random.sample(tr, a.batch)
        idx, mask = encode_train(rows, chunk, stoi, a.block)
        with ctx:
            logits = model(idx, mask)
            loss = sum(F.cross_entropy(logits[h], torch.tensor([classes[h].index(rows[i][h]) for i in chunk], device=DEVICE))
                       for h in HEADS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item()

    best_acc, t0 = -1.0, time.time()
    for it in range(a.iters + 1):
        if it % a.eval_every == 0 or it == a.iters:
            tr_accs, _ = evaluate(model, rows, TRAIN_EVAL, classes, stoi, block=a.block)
            va_accs, confusion = evaluate(model, rows, va, classes, stoi, block=a.block)
            print(f"iter {it:4d} | train acc: "
                  + " | ".join(f"{h} {tr_accs[h]:.3f}" for h in HEADS)
                  + f" | val acc: "
                  + " | ".join(f"{h} {va_accs[h]:.3f}" for h in HEADS)
                  + f" | {time.time()-t0:.0f}s")
            if it > 0 and sum(va_accs.values()) > best_acc:
                best_acc = sum(va_accs.values())
                torch.save({"model": model.state_dict(), "config": cfg, "chars": chars,
                            "classes": classes, "accs": va_accs, "seed": a.seed,
                            "pool": model.pool}, a.out)
        if it == a.iters:
            break
        train_step()

    accs, confusion = evaluate(model, rows, va, classes, stoi, block=a.block)
    tr_accs, _ = evaluate(model, rows, TRAIN_EVAL, classes, stoi, block=a.block)
    print("\n--- val (ostateczna, pełny val) vs train (stała próbka) ---")
    for h in HEADS:
        print(f"{h}: val acc {accs[h]:.4f} | train acc {tr_accs[h]:.4f} "
              f"| top pomyłki: {confusion[h].most_common(5)}")
    print(f"\nbest val (suma acc) zapisane -> {a.out}")

if __name__ == "__main__":
    main()
