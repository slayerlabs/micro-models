"""E_CKA — podobieństwo reprezentacji między niezależnie trenowanymi mikro-modelami.

Linear CKA (Kornblith i in., ICML 2019) + mutual k-NN (główna metryka Platonic, Huh i in., ICML 2024)
na wyjściach bloków, na WSPÓLNYM probe-tekście (znaki obecne we wszystkich słownikach — te same pozycje
dla każdego modelu). Baseline/null: model vs LOSOWY (nietrenowany) o tej samej architekturze.

Pytanie (E_CKA): czy niezależne maluchy zbiegają do wspólnej geometrii (hipoteza platońska na ~0,8M)?
UWAGA: pojedyncza skala NIE falsyfikuje hipotezy platońskiej (ona mówi o trendzie ZE skalą) — to pierwszy
punkt; pełny wynik wymaga sweepu po rozmiarach. Baseline losowy kalibruje „zero".

Użycie: python src/tools/cka.py
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from core.gpt import GPT, GPTConfig

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
DEV = "cpu"

# Probe: REALNY, zróżnicowany fragment korpusu (bez sztucznych duplikatów).
# Naprawa K1: wcześniej był jeden blok powtórzony 3× → fałszywe identyczne sąsiedztwa zawyżały mutual-kNN.
W = 48                          # długość okna (≤ block_size każdego modelu)
PROBE_FILE = "data/jigs.abc"
PROBE_SLICE = (2000, 14000)     # wycinek ze środka korpusu (pomija nagłówki na początku pliku)


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    cfg = ck["config"]
    m = GPT(cfg).to(DEV).eval()
    m.load_state_dict(ck["model"])
    return m, ck["stoi"]


def random_like(cfg, seed=12345):
    torch.manual_seed(seed)
    return GPT(cfg).to(DEV).eval()


@torch.no_grad()
def reps(model, stoi, common):
    """Zwraca listę [N, C] (po jednej macierzy na blok) dla wspólnego probe-tekstu."""
    ids = [stoi[c] for c in common]
    nwin = len(ids) // W
    idx = torch.tensor(ids[: nwin * W], dtype=torch.long).view(nwin, W).to(DEV)
    store = {}
    hooks = [blk.register_forward_hook(
        (lambda i: lambda m, inp, out: store.__setitem__(i, out.detach()))(i)
    ) for i, blk in enumerate(model.blocks)]
    model(idx)
    for h in hooks:
        h.remove()
    return [store[i].reshape(-1, store[i].shape[-1]) for i in range(len(model.blocks))]


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (X.t() @ Y).pow(2).sum()
    den = (X.t() @ X).pow(2).sum().sqrt() * (Y.t() @ Y).pow(2).sum().sqrt()
    return (num / den).item()


def mutual_knn(X, Y, k=10):
    Xn = F.normalize(X, dim=1)
    Yn = F.normalize(Y, dim=1)
    sx = Xn @ Xn.t()
    sy = Yn @ Yn.t()
    sx.fill_diagonal_(-1e9)
    sy.fill_diagonal_(-1e9)
    kx = sx.topk(k, dim=1).indices
    ky = sy.topk(k, dim=1).indices
    inter = sum(len(set(kx[i].tolist()) & set(ky[i].tolist())) for i in range(X.shape[0]))
    return inter / (X.shape[0] * k)


def main():
    paths = {
        "jig": "data/models/jig_ckpt.pt",
        "jig-v2": "data/models/jig_v2_ckpt.pt",
        "waltz": "data/models/waltz_ckpt.pt",
        "reel": "data/models/reel_ckpt.pt",
        "bach": "data/models/bach_ckpt.pt",
        # sweep skali (te same dane, inny seed, różny n_embd) — test PRH: czy CKA rośnie z rozmiarem
        "jig_s1": "data/models/jig_s1_ckpt.pt", "jig_s2": "data/models/jig_s2_ckpt.pt",   # ~0,2M (n_embd 64)
        "jig_l1": "data/models/jig_l1_ckpt.pt", "jig_l2": "data/models/jig_l2_ckpt.pt",   # ~1,8M (n_embd 192)
        # utwardzenie: 3. seed na skalę (error bars) + szerszy zakres (~0,05M, n_embd 32)
        "jig_e32_s1": "data/models/jig_e32_s1_ckpt.pt", "jig_e32_s2": "data/models/jig_e32_s2_ckpt.pt", "jig_e32_s3": "data/models/jig_e32_s3_ckpt.pt",
        "jig_e64_s3": "data/models/jig_e64_s3_ckpt.pt",
        "jig_e128_s3": "data/models/jig_e128_s3_ckpt.pt",
        "jig_e192_s3": "data/models/jig_e192_s3_ckpt.pt",
    }
    models, vocabs = {}, []
    for name, p in paths.items():
        if os.path.exists(p):
            m, stoi = load(p)
            models[name] = (m, stoi)
            vocabs.append(set(stoi))
        else:
            print(f"(pomijam {name}: brak {p})")

    common_set = set.intersection(*vocabs)
    raw = open(PROBE_FILE, encoding="utf-8").read()[PROBE_SLICE[0]:PROBE_SLICE[1]]
    common = "".join(c for c in raw if c in common_set)
    nwin = len(common) // W
    print(f"wspólny słownik: {len(common_set)} znaków | probe REALNY (bez duplikatów): {len(common)} znaków -> N = {nwin} okien x {W} = {nwin*W} próbek\n")

    # baseline losowy o architekturze jiga
    jig_cfg = models["jig"][0].cfg
    models["LOSOWY"] = (random_like(jig_cfg, 12345), models["jig"][1])
    models["LOSOWY2"] = (random_like(jig_cfg, 99999), models["jig"][1])   # właściwy null: losowy vs losowy

    R = {n: reps(m, stoi, common) for n, (m, stoi) in models.items()}
    nlayer = len(R["jig"])
    core = [n for n in ["jig", "jig-v2", "waltz", "reel", "bach", "LOSOWY"] if n in models]

    def pair(metric, a, b):
        return sum(metric(R[a][l], R[b][l]) for l in range(nlayer)) / nlayer

    for title, metric in (("Linear CKA (średnia po blokach)", linear_cka),
                          ("Mutual k-NN (k=10, średnia po blokach)", mutual_knn)):
        print(f"=== {title} ===")
        print("        " + "".join(f"{n:>8}" for n in core))
        for a in core:
            print(f"{a:>8}" + "".join(f"{pair(metric, a, b):>8.3f}" for b in core))
        print()

    # KRZYWA SKALI (utwardzona) — same-task CKA na 4 rozmiarach, mean±std po WSZYSTKICH parach seedów
    print("=== KRZYWA SKALI (utwardzona): same-task CKA, mean±std po parach seedów — test PRH ===")
    SCALES = [
        ("~0,05M", ["jig_e32_s1", "jig_e32_s2", "jig_e32_s3"]),
        ("~0,2M",  ["jig_s1", "jig_s2", "jig_e64_s3"]),
        ("~0,8M",  ["jig", "jig-v2", "jig_e128_s3"]),
        ("~1,8M",  ["jig_l1", "jig_l2", "jig_e192_s3"]),
    ]

    def mean_std(xs):
        m = sum(xs) / len(xs)
        s = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5   # std PRÓBKOWE (/(n-1))
        return m, s

    for label, group in SCALES:
        present = [n for n in group if n in models]
        pairs = list(itertools.combinations(present, 2))
        if not pairs:
            print(f"{label:>8}: (za mało modeli: {present})")
            continue
        cm, cs = mean_std([pair(linear_cka, a, b) for a, b in pairs])
        km, ks = mean_std([pair(mutual_knn, a, b) for a, b in pairs])
        # null PER SKALA: dwa losowe modele architektury z tej skali.
        # (Płaski null liczony tylko na ~0,8M był mylący — fix recenzji 2026-06-23.)
        cfg_s = models[present[0]][0].cfg
        rA = reps(random_like(cfg_s, 12345), models[present[0]][1], common)
        rB = reps(random_like(cfg_s, 99999), models[present[0]][1], common)
        n0 = len(rA)
        cka0 = sum(linear_cka(rA[l], rB[l]) for l in range(n0)) / n0
        knn0 = sum(mutual_knn(rA[l], rB[l]) for l in range(n0)) / n0
        print(f"{label:>8} (n={len(pairs)} par): CKA {cm:.3f}±{cs:.3f} | kNN {km:.3f}±{ks:.3f}  ||  null/skala: CKA {cka0:.3f} | kNN {knn0:.3f}")
    print("PRH przewiduje: podobieństwo ROŚNIE z rozmiarem. Płasko/szum = brak trendu w tym zakresie.\n")

    # kluczowe porównania
    print("=== kluczowe odczyty ===")
    if "jig-v2" in models:
        print(f"jig vs jig-v2 (te same dane, INNY seed):   CKA {pair(linear_cka,'jig','jig-v2'):.3f} | kNN {pair(mutual_knn,'jig','jig-v2'):.3f}")
    print(f"jig vs waltz (inne dane/styl):             CKA {pair(linear_cka,'jig','waltz'):.3f} | kNN {pair(mutual_knn,'jig','waltz'):.3f}")
    print(f"jig vs LOSOWY (trenowany vs losowy):        CKA {pair(linear_cka,'jig','LOSOWY'):.3f} | kNN {pair(mutual_knn,'jig','LOSOWY'):.3f}")
    print(f"NULL właściwy — LOSOWY vs LOSOWY2 (2 losowe): CKA {pair(linear_cka,'LOSOWY','LOSOWY2'):.3f} | kNN {pair(mutual_knn,'LOSOWY','LOSOWY2'):.3f}")
    print("\n=== CKA per-warstwa: jig vs jig-v2 (czy zgodność nie siedzi tylko w L0?) ===")
    for l in range(nlayer):
        print(f"  blok {l}: CKA {linear_cka(R['jig'][l], R['jig-v2'][l]):.3f} | kNN {mutual_knn(R['jig'][l], R['jig-v2'][l]):.3f}")
    print("\nUwaga: jig–jig-v2 = te same dane, inny seed → STABILNOŚĆ względem seeda, nie 'platońska konwergencja'.")


if __name__ == "__main__":
    main()
