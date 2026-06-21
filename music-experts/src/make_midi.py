"""JEDNO POLECENIE: użyj wytrenowanego GPT, by stworzyć nowe melodie + pliki MIDI.

Przykłady:
  python src/make_midi.py                      # 3 melodie w D, do data/out_*.mid
  python src/make_midi.py --key G --n 5         # 5 melodii w G
  python src/make_midi.py --temp 1.0 --out data/eksperyment
"""
import argparse, sys
import torch
from gpt import GPT
from abc_to_midi import to_midi          # konwersja ABC -> MIDI (music21)


def first_tune(raw: str) -> str:
    """Wytnij pierwszą melodię (nagłówek + ciało aż do kolejnego 'X:')."""
    lines = []
    for ln in raw.split("\n"):
        if ln.startswith("X:") and lines:
            break
        lines.append(ln)
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key",  default="D",   help="tonacja, np. D, G, Am, Edor")
    ap.add_argument("--n",    type=int,   default=3,   help="ile melodii")
    ap.add_argument("--temp", type=float, default=0.8, help="temperatura (więcej = śmielej)")
    ap.add_argument("--topk", type=int,   default=20)
    ap.add_argument("--new",  type=int,   default=400, help="ile znaków generować na melodię")
    ap.add_argument("--ckpt", default="data/gpt_ckpt.pt")
    ap.add_argument("--out",  default="data/out", help="prefiks plików wyjściowych")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    stoi, itos, cfg = ck["stoi"], ck["itos"], ck["config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"GPT {model.num_params():,} param | val loss {ck['val_loss']:.3f} | {device}")

    seed = f"X:1\nM:6/8\nK:{args.key}\n"
    made = 0
    for i in range(1, args.n + 1):
        idx = torch.tensor([[stoi[c] for c in seed]], dtype=torch.long, device=device)
        out = model.generate(idx, args.new, temperature=args.temp, top_k=args.topk)[0].tolist()
        tune = first_tune("".join(itos[t] for t in out))
        abc_path, mid_path = f"{args.out}_{i}.abc", f"{args.out}_{i}.mid"
        with open(abc_path, "w", encoding="utf-8") as f:
            f.write(tune + "\n")
        ok = to_midi(tune, mid_path)
        made += int(ok)
        print(f"\n--- #{i}  [{'MIDI OK -> '+mid_path if ok else 'MIDI błąd'}] ---\n{tune}")

    print(f"\nGotowe: {made}/{args.n} plików MIDI ({args.out}_*.mid)")


if __name__ == "__main__":
    main()
