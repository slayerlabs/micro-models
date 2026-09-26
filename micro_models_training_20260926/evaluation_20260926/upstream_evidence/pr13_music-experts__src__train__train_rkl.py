# Autor: Adam Skrodzki
"""On-policy destylacja: KL student↔rotujący nauczyciel (continuation + scratch).

Student generuje rollouty z promptów domeny-nauczyciela (temp/topk jak w benchmarku),
nauczyciel kompetentny w tej domenie ocenia wygenerowane znaki, a strata to per-token
KL między rozkładami studenta i nauczyciela — tylko na WYGENEROWANYCH pozycjach, liczone
w oknach 128, gdzie pierwsze 32 znaki okna to tylko kontekst (nauczyciel
i student przewidują z prawdziwego prefixu, nie z amputowanego środka melodii).
Nauczyciel nigdy nie widzi surowych danych innej domeny niż jego własna — student
uczy się „z drugiej ręki", z samych prawdopodobieństw.

α-mixing: p_mix = (1−α)·p_teacher + α/|V|, |V| = rozmiar wspólnego słownika (54).
Podłoga prawdopodobieństwa α/|V| na każdym znaku — kalibracja na niewytrenowanym
supportie nauczyciela (znaki spoza jego korpusu mają losowe logity po weight-tyingu)
+ ograniczenie gradientów tam, gdzie nauczyciel jest pewny, a student generuje śmieci.
Koszt: lekkie ściągnięcie ku uniform — entropia generacji jest metryką kontrolną.

Wspólny słownik: student i wszyscy nauczyciele muszą mieć identyczny zbiór znaków
(assert). Rotacja PER BATCH: domena (równo jig→reel→waltz) × zadanie (continuation:
prompt o STAŁEJ długości 96 znaków = nagłówki + prefiks ciała; scratch: same nagłówki,
batch z jednego kubełka długości nagłówka). Ewaluacja co eval_interval: KL / entropia /
log-prob nauczyciela na utwalonej puli val per domena i zadanie. Zapisywane oba
checkpointy: best (min val KL — słabo skorelowany z jakością benchmarkową) i last.

Użycie:
  python src/train/train_rkl.py \
    --student data/models/universalist_ckpt.pt \
    --teachers jig=data/models/jig_sh_ckpt.pt reel=data/models/reel_sh_ckpt.pt \
               waltz=data/models/waltz_sh_ckpt.pt \
    --max-iters 4000
"""
import argparse, json, math, os, random, sys, time
from contextlib import nullcontext
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))  # src/tools
import torch
import torch.nn.functional as F
from core.gpt import GPT
from train_judge import load_rows, group_split

MIN_POOL_BODY = 4 * 40   # jak benchmark OOD: ćwiartka prefiksu >= 40 znaków


def build_prompt(row, task, total):
    """continuation: nagłówki + prefiks ciała o STAŁEJ długości total (równa długość
    pozwala generować rollouty batchowo — generate() nie maskuje paddingu).
    scratch: same nagłówki; długość = długość nagłówków (batch z jednego kubełka długości)."""
    head = f"X:1\nM:{row['meter']}\nK:{row['mode']}\n"
    if task == "scratch":
        return head
    return (head + row["body"][:max(1, total - len(head))])[:total]


def window_loss(seq, plen, student, teacher, t2s, alpha, direction, ctx, device, ctx_chars=32):
    """KL per-token na wygenerowanych pozycjach sekwencji (B, L). Okna ze stride: pierwsze
    ctx_chars znaków okna (poza pierwszym) to tylko KONTEKST — nauczyciel i student
    przewidują z prawdziwego prefixu.
    Zwraca (loss, mean_log_p_mix_na_sample, mean_entropia_studenta, n_pozycji).
    Statystyki per POZYCJA per SEKWENCJA (licznik n uwzględnia batch)."""
    B = seq.size(0)
    BLOCK = min(student.cfg.block_size, teacher.cfg.block_size)
    V = student.cfg.vocab_size
    stride = BLOCK - ctx_chars
    loss_sum, tlp_sum, ent_sum, n = 0.0, 0.0, 0.0, 0
    for s in range(0, max(1, seq.size(1) - 1), stride):
        w = seq[:, s:s + BLOCK]
        T = w.size(1)
        if T < 2:
            continue
        lo = ctx_chars if s > 0 else 0            # w 1. oknie kontekstem jest sam prompt
        p = torch.arange(lo, T - 1, device=device)  # lokalne pozycje logitów (predykcja p+1)
        g = s + p + 1                              # globalne indeksy znaków targetowych
        m = g >= plen                              # tylko wygenerowane znaki
        if not bool(m.any()):
            continue
        with torch.no_grad(), ctx:
            tl, _ = teacher(w)
            tp = F.softmax(tl.float(), dim=-1)[:, :, t2s]      # mapowanie na porządek studenta
            pmix = (1 - alpha) * tp + alpha / V
            log_pmix = pmix.clamp_min(1e-12).log()
        with ctx:
            sl, _ = student(w)
            log_ps = sl.float().log_softmax(-1)
            ps = log_ps.exp()
            if direction == "reverse":
                lp = (ps * (log_ps - log_pmix)).sum(-1)        # KL(student || mix)
            else:
                lp = (pmix * (log_pmix - log_ps)).sum(-1)      # KL(mix || student)
            tgt = seq[:, s + 1:s + T]
            lp_m = lp[:, :-1][:, lo:][:, m]
            tlp_m = log_pmix[:, :-1].gather(-1, tgt.unsqueeze(-1))[:, :, 0][:, lo:][:, m]
            ent_m = (-(ps * log_ps).sum(-1))[:, :-1][:, lo:][:, m]
            loss_sum = loss_sum + lp_m.sum()
            tlp_sum = tlp_sum + tlp_m.sum()
            ent_sum = ent_sum + ent_m.sum()
            n = n + m.sum() * B                    # maska wspólna dla całego batcha
    n = max(n, 1)
    return loss_sum / n, tlp_sum / n, ent_sum / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="data/models/universalist_ckpt.pt",
                    help="checkpoint startowy studenta (musi mieć wspólny słownik z nauczycielami)")
    ap.add_argument("--teachers", nargs="+", required=True,
                    help="domena=ckpt, np. jig=data/models/jig_sh_ckpt.pt")
    ap.add_argument("--domains", default="data/models/domains.json")
    ap.add_argument("--csv", default="data/tunes.csv")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--direction", choices=["reverse", "forward"], default="reverse")
    ap.add_argument("--tasks", choices=["continuation", "scratch", "both"], default="both",
                    help="zadania w rotacji (scratch nagowywało regresję scratch w benchmarku)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--new", type=int, default=420)
    ap.add_argument("--prompt-chars", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--out", default="data/models/student_rkl_ckpt.pt")
    ap.add_argument("--last", default="data/models/student_rkl_last.pt",
                    help="ostatni checkpoint (niezależnie od val KL) — do porównania z best")
    ap.add_argument("--losslog", default="data/models/student_rkl_loss.csv")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    print(f"urządzenie: {device} | bf16: {use_bf16} | seed: {a.seed} | "
          f"temp {a.temp} topk {a.topk} new {a.new} | alpha {a.alpha} | {a.direction}")

    # --- student ---
    sck = torch.load(a.student, map_location=device, weights_only=False)
    stoi, itos, scfg = sck["stoi"], sck["itos"], sck["config"]
    student = GPT(scfg); student.load_state_dict(sck["model"]); student.train().to(device)
    V = scfg.vocab_size
    print(f"student: {a.student} | vocab {V} | block {scfg.block_size} | "
          f"parametry {student.num_params():,}")

    # --- nauczyciele: domena -> (model, mapa idx nauczyciel->student) ---
    with open(a.domains, encoding="utf-8") as f:
        domains_meta = json.load(f)
    teachers, t2s_map, meters = {}, {}, {}
    for spec in a.teachers:
        dom, path = spec.split("=", 1)
        tck = torch.load(path, map_location=device, weights_only=False)
        if set(tck["stoi"]) != set(stoi):
            sys.exit(f"nauczyciel {path}: słownik różni się od studenta — wspólny słownik wymagany "
                     f"(brak: {sorted(set(stoi) - set(tck['stoi']))}; "
                     f"dodatkowe: {sorted(set(tck['stoi']) - set(stoi))})")
        t2s_map[dom] = torch.tensor([stoi[tck["itos"][i]] for i in range(len(tck["itos"]))],
                                    device=device)
        tm = GPT(tck["config"]); tm.load_state_dict(tck["model"]); tm.eval().to(device)
        for p in tm.parameters():
            p.requires_grad_(False)
        hit = [d for n, d in domains_meta.items()
               if isinstance(d, dict) and d.get("type") and dom in d["type"].lower()]
        if not hit:
            sys.exit(f"domena '{dom}': brak meter w {a.domains} (wpis type zawierający '{dom}')")
        meters[dom] = hit[0]["meter"]
        teachers[dom] = tm
        print(f"teacher {dom}: {path} | meter {meters[dom]} | block {tck['config'].block_size}")
    order = list(teachers)

    # --- pule promptów: trening z tr, ewaluacja z val (czysta); scratch kubełkuje po
    # długości nagłówka, żeby batch miał równe długości promptów ---
    print(f"czytam {a.csv} ...")
    rows = load_rows(a.csv)
    tr, va = group_split(rows, a.split_seed)
    task_list = ["continuation", "scratch"] if a.tasks == "both" else [a.tasks]
    pools_tr, pools_va, scratch_buckets = {}, {}, {}
    for dom in order:
        ok = lambda i: (dom in rows[i]["type"].lower()
                        and rows[i]["meter"].strip() == meters[dom]
                        and len(rows[i]["body"]) >= MIN_POOL_BODY
                        and all(c in stoi for c in build_prompt(rows[i], "continuation", a.prompt_chars)))
        pools_tr[dom] = [i for i in tr if ok(i)]
        pools_va[dom] = [i for i in va if ok(i)]
        buckets = {}
        for i in pools_tr[dom]:
            buckets.setdefault(len(build_prompt(rows[i], "scratch", a.prompt_chars)), []).append(i)
        scratch_buckets[dom] = buckets
        print(f"  {dom}: pula treningowa {len(pools_tr[dom])}, walidacyjna {len(pools_va[dom])}, "
              f"kubełki scratch: {len(buckets)}")
        if not pools_tr[dom] or not pools_va[dom]:
            sys.exit(f"domena {dom}: pusta pula")
    rng = random.Random(f"{a.seed}|eval")
    eval_ids = {d: rng.sample(pools_va[d], min(a.eval_samples, len(pools_va[d]))) for d in order}

    def rollout_loss(dom, task, ids):
        """(kl, tlp, ent, n) na rolloutach z puli ids. Scratch grupuje po długości nagłówka
        (równe długości promptów w batchu); średnie ważone liczbą pozycji."""
        if task == "continuation":
            chunks = [(ids, a.prompt_chars)]
        else:
            by_len = {}
            for i in ids:
                by_len.setdefault(len(build_prompt(rows[i], "scratch", a.prompt_chars)), []).append(i)
            chunks = [(v, L) for L, v in sorted(by_len.items())]
        kl_s, tlp_s, ent_s, n_s = 0.0, 0.0, 0.0, 0
        for chunk, plen in chunks:
            enc = torch.tensor([[stoi[c] for c in build_prompt(rows[i], task, a.prompt_chars)]
                                for i in chunk], device=device)
            gen = student.generate(enc, a.new, temperature=a.temp, top_k=a.topk)
            kl, tlp, ent, n = window_loss(gen, plen, student, teachers[dom],
                                          t2s_map[dom], a.alpha, a.direction, ctx, device)
            kl_s += float(kl) * n; tlp_s += float(tlp) * n; ent_s += float(ent) * n; n_s += n
        if n_s == 0:
            return 0.0, 0.0, 0.0, 1
        return kl_s / n_s, tlp_s / n_s, ent_s / n_s, n_s

    @torch.no_grad()
    def evaluate():
        """KL / log-prob nauczyciela / entropia studenta na utwalonych val promptach,
        per domena i per zadanie."""
        out = {}
        for dom in order:
            student.eval()
            for task in task_list:
                out[(dom, task)] = rollout_loss(dom, task, eval_ids[dom])
            student.train()
        return out

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.1)

    def lr_at(it):
        warmup = 100
        if it < warmup:
            return a.lr * it / warmup
        r = (it - warmup) / max(1, a.max_iters - warmup)
        return a.lr * 0.1 + 0.5 * a.lr * 0.9 * (1 + math.cos(math.pi * r))

    log = [("iter", "train_loss") +
           tuple(c for d in order for t in task_list
                 for c in (f"kl_{d}_{t}", f"tlp_{d}_{t}", f"ent_{d}_{t}"))]
    best_kl, t0 = float("inf"), time.time()
    for it in range(a.max_iters + 1):
        if it % a.eval_interval == 0 or it == a.max_iters:
            ev = evaluate()
            avg = sum(v[0] for v in ev.values()) / len(ev)
            row = [it, ""] + [c for d in order for t in task_list for c in ev[(d, t)]]
            log.append(tuple(row))
            line = " | ".join(f"{d}/{t[:4]} kl {ev[(d, t)][0]:.3f} tlp {ev[(d, t)][1]:.3f} "
                              f"ent {ev[(d, t)][2]:.3f}" for d in order for t in task_list)
            print(f"iter {it:4d} | val KL avg {avg:.3f} | {line} | {time.time()-t0:.0f}s")
            if avg < best_kl:
                best_kl = avg
                student.eval()
                torch.save({"model": student.state_dict(), "config": scfg,
                            "stoi": stoi, "itos": itos, "val_loss": best_kl}, a.out)
                student.train()
        if it == a.max_iters:
            student.eval()
            torch.save({"model": student.state_dict(), "config": scfg,
                        "stoi": stoi, "itos": itos, "val_loss": avg}, a.last)
            student.train()
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        dom = order[it % len(order)]                     # rotacja per batch
        task = task_list[(it // len(order)) % len(task_list)]   # rotacja zadań nad domenami
        if task == "continuation":
            ids = [random.choice(pools_tr[dom]) for _ in range(a.batch_size)]
            plen = a.prompt_chars
        else:                                            # scratch: batch z jednego kubełka długości
            plen = random.choice(sorted(scratch_buckets[dom]))
            ids = [random.choice(scratch_buckets[dom][plen]) for _ in range(a.batch_size)]
        enc = torch.tensor([[stoi[c] for c in build_prompt(rows[i], task, a.prompt_chars)]
                            for i in ids], device=device)
        student.eval()                                   # rollouty bez dropoutu (jak w benchmarku)
        with torch.no_grad(), ctx:
            seq = student.generate(enc, a.new, temperature=a.temp, top_k=a.topk)
        student.train()
        loss, _, _, _ = window_loss(seq, plen, student, teachers[dom],
                                    t2s_map[dom], a.alpha, a.direction, ctx, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if it % 20 == 0:
            print(f"  it {it:4d} [{dom:5s}/{task[:4]}] loss {float(loss.detach()):.4f} | {time.time()-t0:.0f}s")

    with open(a.losslog, "w", encoding="utf-8") as f:
        f.write("\n".join(",".join(str(x) for x in r) for r in log))
    print(f"\ngotowe. best val KL: {best_kl:.3f}")
    print(f"checkpoint best -> {a.out} | last -> {a.last} | krzywa -> {a.losslog}")


if __name__ == "__main__":
    main()
