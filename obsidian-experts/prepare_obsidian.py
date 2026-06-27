"""prepare_obsidian.py — dekompozytor strukturalny ObsidianVault -> korpus mikro-LLM (podejście A).

Bierze notatki dialektyczne {cel, teza, antyteza, synteza, decyzja, fakt, powod}, rozbija każdą
DETERMINISTYCZNIE wg schematu AST-T2 (SEDNO=Teza, ARGUMENTACJA=body, ZALOZENIA=kontekst,
SYNTEZA=rozstrzygnięcie, LINKI=[[...]]) i zapisuje TRZY korpusy do lab/data/ na potrzeby
ablacji decomp-vs-raw + uczciwego eval bits/char na wspólnym held-out:

  korpus-obsidian.txt      — notatki TRAIN zdekomponowane (z markerami ### TYP/SEDNO/...)
  korpus-obsidian-raw.txt  — notatki TRAIN surowe (sam clean_text body)
  heldout-raw.txt          — notatki HELD-OUT surowe (wspólny target eval dla WSZYSTKICH modeli)

Split 90/10 po CAŁYCH notatkach (seed-deterministyczny). NIE używa collect_nodes (truncate 1000 zn.).
Odwzorowuje regexy validate-vault.ps1 (frontmatter, wikilink, wykluczenia) + dokłada: strip BOM,
tolerancja wciętego ---, klucz (project,id) bo id nie jest globalnie unikalne.

Użycie:
  python prepare_obsidian.py
  python prepare_obsidian.py --vault C:\\ProjektyPublic\\ObsidianVault --min-chars 200
"""
import argparse, os, re, sys, random
from collections import Counter

# clean_text (vendored z Dendrometria/dendro_eval.py) — usuwa [[ ]], #*>`, zwija whitespace.
def clean_text(body):
    s = re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", r"\1", body)
    s = re.sub(r"[#*>`]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

HERE = os.path.dirname(os.path.abspath(__file__))

DIALECTIC = {"cel", "teza", "antyteza", "synteza", "decyzja", "fakt", "powod"}
EXCLUDE_DIRS = {".git", ".obsidian", ".trash", "scripts", "archive"}
ENTRY_POINTS = {"README", "INDEX", "AGENTS", "CONTRIBUTING", "CLAUDE", "LICENSE"}

# Fallback: prefiks folderu-roli -> typ (gdy frontmatter nie ma 'type'). "20-powody" PRZED "20".
FOLDER_TYPE = [
    ("20-powody", "powod"),
    ("00", "cel"), ("10", "teza"), ("15", "antyteza"),
    ("25", "synteza"), ("20", "decyzja"), ("99", "decyzja"), ("30", "fakt"),
]

# Tolerancyjny frontmatter: BOM + wcięty ---/CRLF (szerzej niż validate-vault.ps1).
FM_RE = re.compile(r"^\ufeff?\s*---\s*\r?\n(.*?)\r?\n\s*---\s*\r?\n", re.S)
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+):\s*(.*)$")
IDS_RE = re.compile(r'"([^"]+)"')                                   # listy flow ["a","b"]
WIKILINK_RE = re.compile(r'\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]')
IMG_EXT = (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp")
EXTERNAL_RE = re.compile(r'(IQSteel|ProjektyPublic|src|docs)')
ROLE_PREFIX_RE = re.compile(r'^\*\*[^*]+\*\*\s*[:\-—]\s*')           # "> **Teza**: ..."
QUOTE_CHARS = ' "\u201e\u201d\u201c\u00ab\u00bb'

SEDNO_PREFIXES = ("teza", "antyteza", "synteza", "cel", "stwierdzenie", "twierdzenie", "decyzja")
ZALOZENIA_PREFIXES = ("kontekst", "po co", "definicja", "dlaczego to nie", "założenia", "zalozenia", "problem")
SYNTEZA_PREFIXES = ("synteza", "wynik dialektyki", "rozstrzygnięcie", "rozstrzygniecie", "kryterium", "decyzja", "uzasadnienie")
RELATION_FIELDS = ("parents", "synthesizes", "synthesizes_with", "contradicts")


def parse_fm(text):
    """(fields, body). Tolerancyjny na BOM/wcięty ---/CRLF."""
    m = FM_RE.match(text)
    if not m:
        return {}, text.lstrip("\ufeff")
    fields = {}
    for line in m.group(1).splitlines():
        fm = FIELD_RE.match(line)
        if fm:
            fields[fm.group(1).strip()] = fm.group(2).strip()
    return fields, text[m.end():]


def classify(fm, rel_path):
    """type z frontmatter = autorytet; gdy brak -> fallback z prefiksu folderu-roli."""
    t = (fm.get("type") or "").strip().strip('"').lower()
    if t:
        return t
    for seg in rel_path.replace("\\", "/").split("/"):
        segl = seg.lower()
        for pre, typ in FOLDER_TYPE:
            if segl.startswith(pre):
                return typ
    return ""


def header_matches(title, prefixes):
    t = title.strip().lower().lstrip("#").strip()
    return any(t.startswith(p) for p in prefixes)


def first_paragraph(text):
    paras = re.split(r"\n\s*\n", text.strip())
    return paras[0] if paras else ""


def lead_blockquote(lines):
    """Pierwszy blockquote po H1 -> (tekst, (start,end)) lub (None, None)."""
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s == "" or s == "---" or (s.startswith("# ") and not s.startswith("## ")):
            i += 1
            continue
        break
    if i < n and lines[i].lstrip().startswith(">"):
        j = i
        while j < n and lines[j].lstrip().startswith(">"):
            j += 1
        text = " ".join(lines[k].lstrip().lstrip(">").strip() for k in range(i, j) if lines[k].strip())
        return text, (i, j)
    return None, None


def level2_sections(lines):
    """Sekcje poziomu '## ' jako {title, start, end} (end exclusive; H1 też zamyka)."""
    n = len(lines)
    bounds = [i for i, l in enumerate(lines)
              if l.lstrip().startswith("## ") or (l.lstrip().startswith("# ") and not l.lstrip().startswith("## "))]
    secs = []
    for k, b in enumerate(bounds):
        ls = lines[b].lstrip()
        if not ls.startswith("## "):
            continue
        end = bounds[k + 1] if k + 1 < len(bounds) else n
        secs.append({"title": ls[3:].strip(), "start": b, "end": end,
                     "content": "\n".join(lines[b + 1:end])})
    return secs


def clean_sedno(bq_text):
    s = ROLE_PREFIX_RE.sub("", bq_text.strip()).strip(QUOTE_CHARS).strip()
    return clean_text(s)


def extract_links(body, fm):
    """Deduped cele wikilinków z body + relacje z frontmatter. Skip image-embedy + external repo."""
    out, seen = [], set()

    def add(t):
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    for m in WIKILINK_RE.finditer(body):
        raw = m.group(1).strip().rstrip("\\").strip()
        is_embed = m.start() > 0 and body[m.start() - 1] == "!"
        if is_embed and raw.lower().endswith(IMG_EXT):
            continue                                                # embed obrazka/pdf -> pomiń
        if (raw.startswith("../") or raw.startswith("..\\")) and EXTERNAL_RE.search(raw):
            continue                                                # link do innego repo/vaulta
        base = re.split(r"[\\/]", raw)[-1]
        base = re.sub(r"\.md$", "", base, flags=re.I).strip()
        add(base)
    for rel in RELATION_FIELDS:
        for v in IDS_RE.findall(fm.get(rel, "")):
            add(v.strip())
    return out


def build_argumentacja(body, bq_span):
    """CAŁE body PO usunięciu: H1, lead-blockquote, sekcji ## Powiązania -> clean_text (lossless)."""
    lines = body.split("\n")
    n = len(lines)
    keep = [True] * n
    for i, l in enumerate(lines):
        ls = l.lstrip()
        if ls.startswith("# ") and not ls.startswith("## "):
            keep[i] = False
    if bq_span:
        for i in range(bq_span[0], bq_span[1]):
            keep[i] = False
    for sec in level2_sections(lines):
        if header_matches(sec["title"], ("powiązania", "powiazania")):
            for i in range(sec["start"], sec["end"]):
                keep[i] = False
    return clean_text("\n".join(l for i, l in enumerate(lines) if keep[i]))


def decompose(fm, body, ntype, title):
    """Rekord z markerami. ARGUMENTACJA zawsze; reszta opcjonalna (pomijana gdy pusta)."""
    lines = body.split("\n")
    bq_text, bq_span = lead_blockquote(lines)
    secs = level2_sections(lines)

    def first_section(prefixes):
        for s in secs:
            if header_matches(s["title"], prefixes):
                return clean_text(first_paragraph(s["content"]))
        return ""

    # SEDNO
    if bq_text:
        sedno = clean_sedno(bq_text)
    else:
        sedno = first_section(SEDNO_PREFIXES)
    # ZALOZENIA = parents (id-y) + sekcja kontekstowa
    parents = " ".join(IDS_RE.findall(fm.get("parents", "")))
    ctx = first_section(ZALOZENIA_PREFIXES)
    zalozenia = (("rodzice: " + parents + ". " if parents else "") + ctx).strip()
    # SYNTEZA
    synteza = first_section(SYNTEZA_PREFIXES)
    # ARGUMENTACJA (lossless)
    argumentacja = build_argumentacja(body, bq_span)
    # LINKI
    linki = ", ".join(extract_links(body, fm))

    rec = [f"### TYP: {ntype}", f"### TYTUL: {title}"]
    if sedno:
        rec.append(f"### SEDNO: {sedno}")
    if zalozenia:
        rec.append(f"### ZALOZENIA: {zalozenia}")
    rec.append(f"### ARGUMENTACJA: {argumentacja}")
    if synteza:
        rec.append(f"### SYNTEZA: {synteza}")
    if linki:
        rec.append(f"### LINKI: {linki}")
    return "\n".join(rec)


def collect(vault, min_chars):
    notes = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for f in files:
            if not f.endswith(".md"):
                continue
            stem = os.path.splitext(f)[0]
            if stem.upper() in ENTRY_POINTS:
                continue
            path = os.path.join(root, f)
            try:
                text = open(path, encoding="utf-8").read()
            except Exception:
                continue
            rel = os.path.relpath(path, vault)
            fm, body = parse_fm(text)
            ntype = classify(fm, rel)
            if ntype not in DIALECTIC:
                continue
            raw = clean_text(body)
            if len(raw) < min_chars:
                continue
            title = (fm.get("title") or "").strip().strip('"') or stem
            nid = (fm.get("id") or "").strip().strip('"') or stem.split("-")[0]
            project = rel.replace("\\", "/").split("/")[0]
            notes.append({
                "key": (project, nid), "project": project, "path": rel, "type": ntype,
                "raw": raw, "decomp": decompose(fm, body, ntype, title),
            })
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=r"C:\ProjektyPublic\ObsidianVault")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--lab", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    notes = collect(args.vault, args.min_chars)
    notes.sort(key=lambda x: (x["project"], x["path"]))          # determinizm przed shuffle
    random.Random(args.seed).shuffle(notes)
    n_train = int(0.9 * len(notes))
    train, held = notes[:n_train], notes[n_train:]

    os.makedirs(args.lab, exist_ok=True)
    p_decomp = os.path.join(args.lab, "korpus-obsidian.txt")
    p_raw = os.path.join(args.lab, "korpus-obsidian-raw.txt")
    p_held = os.path.join(args.lab, "heldout-raw.txt")
    open(p_decomp, "w", encoding="utf-8").write("\n\n".join(n["decomp"] for n in train))
    open(p_raw, "w", encoding="utf-8").write("\n\n".join(n["raw"] for n in train))
    open(p_held, "w", encoding="utf-8").write("\n\n".join(n["raw"] for n in held))

    def chars(p):
        return len(open(p, encoding="utf-8").read())

    print(f"vault: {args.vault} | seed {args.seed} | min-chars {args.min_chars}")
    print(f"notatki dialektyczne: {len(notes)} -> train {len(train)} / held-out {len(held)}")
    print(f"  {p_decomp}: {chars(p_decomp):,} zn.")
    print(f"  {p_raw}: {chars(p_raw):,} zn.")
    print(f"  {p_held}: {chars(p_held):,} zn.")
    dist_tr = Counter(n["type"] for n in train)
    dist_he = Counter(n["type"] for n in held)
    print("rozkład typów (train | held):")
    for t in sorted(DIALECTIC):
        print(f"  {t:9s}: {dist_tr.get(t,0):4d} | {dist_he.get(t,0):3d}")


if __name__ == "__main__":
    main()
