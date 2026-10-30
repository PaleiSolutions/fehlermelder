#!/usr/bin/env python3
"""
Interner Fehler-Melder: Sagen die Hidden States eines offenen LLM vorher, ob seine gerade
gegebene Antwort falsch ist – besser als (a) verbalisierte Sicherheit und (b) Token-Log-Prob?

Zwei Phasen:
  collect  – Modell beantwortet jede Frage (greedy), Antwort wird gegen Aliase gelabelt,
             Hidden States am letzten Antwort-Token (alle Schichten) und Token-Log-Probs werden
             aus einem Teacher-Forcing-Pass gelesen; separater Call fragt nach Sicherheit 0–100.
             Ergebnis wird gecacht (outputs/cache_<modell>.npz + generations.csv).
  analyze  – Split train/val/test (stratifiziert, fixer Seed), Schicht + Regularisierung nur per
             Validierung gewählt, Test-Split genau einmal ausgewertet. Bootstrap-CIs, gepaarte
             Differenzen, Wiederholung über viele Splits (Robustheit), Eskalationskurve, Beispiele.

Aufruf:
  python script.py                                   # Qwen2.5-7B-Instruct, alles
  python script.py --model Qwen/Qwen2.5-3B-Instruct  # leichter Fallback
  python script.py --phase analyze                   # nur Auswertung aus dem Cache
"""
import argparse
import json
import math
import os
import re
import time
import unicodedata

import numpy as np

# --------------------------------------------------------------------------------------
# Antwort-Normalisierung und Matching
# --------------------------------------------------------------------------------------
ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> list:
    """Lowercase, Akzente entfernen, Interpunktion weg (Dezimalpunkt in Zahlen bleibt),
    Tausendertrenner weg, Artikel weg. Rückgabe: Token-Liste."""
    s = s.replace("€", "e").replace("&", " and ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)          # 1,000 -> 1000
    s = re.sub(r"(?<=\d)\.(?=\d)", "\x00", s)           # 4.5 bleibt 4.5
    s = re.sub(r"[^a-z0-9\x00]+", " ", s).replace("\x00", ".")
    toks = [t for t in s.split() if t not in ARTICLES]
    # leichte Plural-Normalisierung ("asset"/"assets", "requirement"/"requirements"),
    # symmetrisch auf Antwort und Alias angewandt
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t[0].isdigit() else t for t in toks]


def contains_subseq(hay: list, needle: list) -> bool:
    n = len(needle)
    if n == 0:
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def is_correct(pred: str, aliases: list) -> bool:
    """Korrekt, wenn ein normalisierter Alias als zusammenhängende Token-Folge in der
    normalisierten (erste Zeile, kurze) Antwort vorkommt. Standard bei TriviaQA-artigen
    Auswertungen; Grenzen siehe README (z. B. Aufzählungen mehrerer Kandidaten)."""
    p = normalize(pred)
    return any(contains_subseq(p, normalize(a)) for a in aliases)


# --------------------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------------------
def load_items(source: str, n: int, seed: int):
    from questions import load_inline
    if source == "triviaqa":
        try:
            from datasets import load_dataset
            ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(ds), size=min(n or 400, len(ds)), replace=False)
            items = []
            for k, i in enumerate(idx):
                r = ds[int(i)]
                items.append({"id": f"tqa{k:04d}", "triviaqa_question_id": r["question_id"],
                              "tier": "?", "domain": "triviaqa",
                              "question": r["question"],
                              # kanonische Antwort zuerst (wird in examples.csv als "gold" gezeigt)
                              "aliases": list(dict.fromkeys([r["answer"]["value"]] +
                                                            r["answer"]["aliases"]))})
            print(f"[data] TriviaQA: {len(items)} Fragen")
            return items
        except Exception as e:  # noqa: BLE001
            print(f"[data] TriviaQA nicht erreichbar ({e!r}) – nutze Inline-Datensatz.")
    items = load_inline()
    if n:
        rng = np.random.default_rng(seed)
        items = [items[i] for i in sorted(rng.choice(len(items), size=min(n, len(items)),
                                                      replace=False))]
    print(f"[data] Inline: {len(items)} Fragen")
    return items


# --------------------------------------------------------------------------------------
# Phase 1: Sammeln
# --------------------------------------------------------------------------------------
SYSTEM_ANSWER = ("You are a knowledgeable assistant. Answer the question with only the short "
                 "answer (a name, number, date or a few words). Do not explain.")
CONF_PROMPT = ("How confident are you that your answer is correct? Reply with a single integer "
               "from 0 to 100 and nothing else.")


def pick_device(pref: str):
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, pref: str):
    import torch
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if pref != "auto":
        return table[pref]
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16   # bf16 auf MPS je nach macOS/torch langsam oder nicht verfügbar
    return torch.float32


def load_model(name: str, device: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # ältere transformers-Versionen
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device).eval()
    return tok, model


def chat_ids(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def greedy(model, tok, input_ids, max_new_tokens):
    import torch
    with torch.inference_mode():
        out = model.generate(input_ids.to(model.device),
                             attention_mask=torch.ones_like(input_ids).to(model.device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return out[0, input_ids.shape[1]:].tolist()


def trim_answer(tok, gen_ids):
    """Bis EOS/Turn-Ende und bis zur ersten Zeilenumbruch-Position kürzen.
    Die so gekürzten Token sind exakt das, worauf gelabelt und gemessen wird."""
    stop = set(x for x in [tok.eos_token_id, tok.pad_token_id] if x is not None)
    for t in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id:
            stop.add(tid)
    ids = []
    for t in gen_ids:
        if t in stop:
            break
        ids.append(t)
        if "\n" in tok.decode(ids):
            ids.pop()
            break
    # führende/trailing Leer-Token sind unkritisch; Text wird für Label gestrippt
    return ids


def collect(args):
    import torch
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    print(f"[model] {args.model} auf {device} ({dtype})")
    tok, model = load_model(args.model, device, dtype)
    items = load_items(args.dataset, args.limit, args.seed)

    H, rows = [], []
    t0 = time.time()
    for k, it in enumerate(items):
        msgs = [{"role": "system", "content": SYSTEM_ANSWER},
                {"role": "user", "content": it["question"]}]
        p_ids = chat_ids(tok, msgs)
        gen = greedy(model, tok, p_ids, args.max_new_tokens)
        a_ids = trim_answer(tok, gen)
        answer = tok.decode(a_ids, skip_special_tokens=True).strip()
        if len(a_ids) == 0 or not answer:
            rows.append({**it, "answer": answer, "skipped": True})
            continue

        # Teacher-Forcing-Pass über Prompt + exakt die generierten Antwort-Token:
        # Hidden States aller Schichten am letzten Antwort-Token + Log-Probs der Antwort-Token.
        full = torch.cat([p_ids, torch.tensor([a_ids])], dim=1).to(model.device)
        with torch.inference_mode():
            out = model(full, output_hidden_states=True)
        P, T = p_ids.shape[1], full.shape[1]
        hs = torch.stack([h[0, T - 1, :] for h in out.hidden_states]).float().cpu().numpy()
        logits = out.logits[0, P - 1:T - 1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = full[0, P:T]
        tok_lp = lp.gather(1, tgt[:, None])[:, 0].cpu().numpy()

        # Verbalisierte Sicherheit: separater Generierungs-Call, gleiche Konversation fortgesetzt.
        cmsgs = msgs + [{"role": "assistant", "content": answer},
                        {"role": "user", "content": CONF_PROMPT}]
        c_gen = greedy(model, tok, chat_ids(tok, cmsgs), 8)
        c_txt = tok.decode(c_gen, skip_special_tokens=True).strip()
        m = re.search(r"\d{1,3}(?:\.\d+)?", c_txt)
        conf = float(min(100.0, max(0.0, float(m.group())))) if m else float("nan")

        H.append(hs.astype(np.float16))
        rows.append({**it, "answer": answer, "skipped": False,
                     "correct": bool(is_correct(answer, it["aliases"])),
                     "mean_logprob": float(tok_lp.mean()), "min_logprob": float(tok_lp.min()),
                     "n_answer_tokens": int(len(a_ids)),
                     "verbal_raw": c_txt, "verbal_conf": conf})
        if (k + 1) % 10 == 0 or k == len(items) - 1:
            done = [r for r in rows if not r["skipped"]]
            acc = np.mean([r["correct"] for r in done]) if done else float("nan")
            el = time.time() - t0
            print(f"  {k+1}/{len(items)}  acc={acc:.2f}  {el/(k+1):.1f}s/Frage  "
                  f"ETA {el/(k+1)*(len(items)-k-1)/60:.1f} min", flush=True)

    os.makedirs(args.out, exist_ok=True)
    kept = [r for r in rows if not r["skipped"]]
    np.savez_compressed(cache_path(args), H=np.stack(H),
                        meta=np.array(json.dumps({"model": args.model, "rows": kept})))
    import pandas as pd
    df = pd.DataFrame(rows)
    df["aliases"] = df["aliases"].apply(lambda a: " ; ".join(a))
    df.to_csv(os.path.join(args.out, "generations.csv"), index=False)
    print(f"[collect] {len(kept)} Antworten gecacht, {len(rows)-len(kept)} leer übersprungen.")


def cache_path(args):
    return os.path.join(args.out, f"cache_{args.model.replace('/', '__')}.npz")


# --------------------------------------------------------------------------------------
# Phase 2: Auswerten
# --------------------------------------------------------------------------------------
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fit_probe(X, y, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                           random_state=seed))
    return clf.fit(X, y)


def split(y, seed, frac=(0.6, 0.2, 0.2)):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=frac[0], stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=frac[1] / (frac[1] + frac[2]),
                              stratify=y[rest], random_state=seed)
    return tr, va, te


def select_and_test(H, y, tr, va, te, seed):
    """Schicht + C ausschließlich auf Validierung wählen; danach Fit auf train+val, Test einmal."""
    from sklearn.metrics import roc_auc_score
    from joblib import Parallel, delayed
    L = H.shape[1]

    def one_layer(l):
        return [roc_auc_score(y[va], fit_probe(H[tr, l], y[tr], C, seed)
                              .predict_proba(H[va, l])[:, 1]) for C in C_GRID]

    val = np.full((L, len(C_GRID)), np.nan)
    # Schicht 0 = Embedding-Ausgabe (nicht kontextualisiert) wird ausgelassen
    for l, r in zip(range(1, L), Parallel(n_jobs=-1)(delayed(one_layer)(l) for l in range(1, L))):
        val[l] = r
    l_best, j_best = np.unravel_index(np.nanargmax(val), val.shape)
    trva = np.concatenate([tr, va])
    clf = fit_probe(H[trva, l_best], y[trva], C_GRID[j_best], seed)
    p_test = clf.predict_proba(H[te, l_best])[:, 1]
    return int(l_best), float(C_GRID[j_best]), val, p_test


def bootstrap(y, scores: dict, n=2000, seed=0):
    """Perzentil-Bootstrap auf dem Test-Split, gepaart über Methoden."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    res = {k: [] for k in scores}
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() == y[b].max():
            continue
        for k, s in scores.items():
            res[k].append(roc_auc_score(y[b], s[b]))
    return {k: np.array(v) for k, v in res.items()}


def escalation(y_wrong, score, fracs, seed=0):
    """Anteil aller Fehler, die gefangen werden, wenn die Top-x % nach score eskaliert werden.
    Gleichstände (z. B. viele '100' bei verbaler Sicherheit) werden zufällig aufgelöst –
    nicht zugunsten der Methode."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))
    ys = y_wrong[order]
    tot = ys.sum()
    out = []
    for f in fracs:
        k = int(round(f * len(ys)))
        out.append(ys[:k].sum() / tot if tot else float("nan"))
    return np.array(out)


def analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    z = np.load(cache_path(args), allow_pickle=False)
    H = z["H"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    df = pd.DataFrame(meta["rows"])
    # Labels hier neu berechnen, damit Matcher-Korrekturen ohne neue Generierung greifen
    relabel = df.apply(lambda r: is_correct(r["answer"], r["aliases"]), axis=1)
    if (relabel != df["correct"]).any():
        print(f"  Labels neu berechnet: {int((relabel != df['correct']).sum())} geändert")
    df["correct"] = relabel
    g = os.path.join(args.out, "generations.csv")
    if os.path.exists(g):
        gen = pd.read_csv(g)
        gen.loc[~gen["skipped"].astype(bool), "correct"] = relabel.to_numpy()
        gen.to_csv(g, index=False)
    y =(~df["correct"].astype(bool)).to_numpy().astype(int)        # 1 = FALSCH (Zielklasse)
    N, L = len(y), H.shape[1]
    acc = 1 - y.mean()
    print(f"[analyze] {meta['model']}: N={N}, Genauigkeit={acc:.3f}, Schichten={L-1}")
    if not 0.3 <= acc <= 0.8:
        print("  WARNUNG: Klassenbalance außerhalb 30–80 % – AUROC-Schätzer werden instabil.")

    verbal = df["verbal_conf"].to_numpy(dtype=float)
    parse_rate = float(np.isfinite(verbal).mean())
    verbal_f = np.where(np.isfinite(verbal), verbal, np.nanmedian(verbal))  # Imputation: Median
    # Baseline-Scores als "Fehler-Score" (höher = eher falsch). Kein Training nötig.
    base = {"verbal": -verbal_f, "logprob_mean": -df["mean_logprob"].to_numpy(),
            "logprob_min": -df["min_logprob"].to_numpy()}

    # ---------- Haupt-Split ----------
    tr, va, te = split(y, args.seed)
    l_best, C_best, val_grid, p_test = select_and_test(H, y, tr, va, te, args.seed)
    print(f"  beste Schicht (Val): {l_best}/{L-1}, C={C_best}, Val-AUROC={np.nanmax(val_grid):.3f}")
    S = {"probe": p_test, **{k: v[te] for k, v in base.items()}}
    auc = {k: float(roc_auc_score(y[te], s)) for k, s in S.items()}
    bs = bootstrap(y[te], S, n=args.n_boot, seed=args.seed)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    diffs = {}
    for k in base:
        d = bs["probe"] - bs[k]
        diffs[f"probe_minus_{k}"] = {"mean": float(d.mean()),
                                     "ci95": [float(np.percentile(d, 2.5)),
                                              float(np.percentile(d, 97.5))],
                                     "p_boot_le_0": float((d <= 0).mean())}
    for k in S:
        print(f"  Test-AUROC {k:13s} {auc[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")

    # ---------- Robustheit: viele Splits, jeweils eigene Val-Auswahl ----------
    rob = {k: [] for k in S}
    rob_layers, rob_esc = [], {k: [] for k in S}
    for s in range(args.n_repeats):
        tr2, va2, te2 = split(y, 1000 + s)
        lb, _, _, pt = select_and_test(H, y, tr2, va2, te2, 1000 + s)
        rob_layers.append(lb)
        sc = {"probe": pt, **{k: v[te2] for k, v in base.items()}}
        for k, v in sc.items():
            rob[k].append(roc_auc_score(y[te2], v))
            rob_esc[k].append(escalation(y[te2], v, [0.2], seed=s)[0])
        print(f"  Robustheit {s+1}/{args.n_repeats}: probe={rob['probe'][-1]:.3f} "
              f"logprob={rob['logprob_mean'][-1]:.3f} verbal={rob['verbal'][-1]:.3f} (L{lb})",
              flush=True)
    rob_sum = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
               for k, v in rob.items()}
    win = {f"probe_beats_{k}": float(np.mean(np.array(rob["probe"]) > np.array(rob[k])))
           for k in base}

    # ---------- Plots ----------
    os.makedirs(args.out, exist_ok=True)
    names = {"probe": f"Hidden-State-Probe (Schicht {l_best})", "verbal": "Verbalisierte Sicherheit",
             "logprob_mean": "Mittlerer Token-Log-Prob", "logprob_min": "Minimaler Token-Log-Prob"}
    colors = {"probe": "#2a6fdb", "verbal": "#d9822b", "logprob_mean": "#3a9d5d",
              "logprob_min": "#9b9b9b"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, s in S.items():
        fpr, tpr, _ = roc_curve(y[te], s)
        ax[0].plot(fpr, tpr, color=colors[k], lw=2.2 if k == "probe" else 1.6,
                   ls="--" if k == "logprob_min" else "-",
                   label=f"{names[k]}: {auc[k]:.2f} [{ci[k][0]:.2f}–{ci[k][1]:.2f}]")
    ax[0].plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls=":")
    ax[0].set(xlabel="Falsch-Alarm-Rate (richtige Antworten markiert)",
              ylabel="Trefferquote (falsche Antworten erkannt)",
              title=f"Fehlererkennung, Test-Split (n={len(te)}, {int(y[te].sum())} Fehler)")
    ax[0].legend(fontsize=7.5, loc="lower right")
    vb = np.r_[np.nan, val_grid[1:].max(axis=1)]
    ax[1].plot(range(1, L), vb[1:], color=colors["probe"], marker="o", ms=3)
    ax[1].axvline(l_best, color="#888", ls=":", lw=1)
    ax[1].set(xlabel="Schicht", ylabel="Val-AUROC (bestes C)", title="Schichtwahl auf Validierung")
    fig.suptitle(meta["model"], fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "roc.png"), dpi=150)
    plt.close(fig)

    fracs = np.linspace(0, 1, 101)
    err_rate = y[te].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    esc = {}
    for k in ["probe", "logprob_mean", "verbal"]:
        esc[k] = escalation(y[te], S[k], fracs, seed=args.seed)
        ax.plot(fracs * 100, esc[k] * 100, color=colors[k], lw=2.2 if k == "probe" else 1.6,
                label=names[k])
    ax.plot(fracs * 100, fracs * 100, color="#bbbbbb", ls=":", label="Zufällig")
    ax.plot(fracs * 100, np.minimum(fracs / err_rate, 1) * 100, color="#555", ls="--", lw=1,
            label="Orakel")
    ax.axvline(20, color="#ddd", lw=1)
    ax.set(xlabel="Anteil an Menschen eskalierter Antworten (%)",
           ylabel="Anteil aller Fehler, der abgefangen wird (%)",
           title=f"Eskalationskurve, Test-Split (Fehlerrate {err_rate:.0%})", xlim=(0, 100), ylim=(0, 102))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "escalation.png"), dpi=150)
    plt.close(fig)

    # ---------- Qualitative Beispiele ----------
    t = df.iloc[te].copy()
    t["probe_p_wrong"] = p_test
    t["probe_rank_pct"] = pd.Series(p_test).rank(pct=True).to_numpy() * 100
    wrong = t[~t["correct"].astype(bool)]
    confident = wrong["verbal_conf"].fillna(0) >= 80
    caught = wrong[confident & (wrong["probe_p_wrong"] >= 0.5)].sort_values("probe_p_wrong",
                                                                             ascending=False)
    caught = caught.head(5).assign(kind="souveraen_falsch_von_probe_markiert")
    missed = wrong[confident & (wrong["probe_p_wrong"] < 0.5)].sort_values("probe_p_wrong")
    missed = missed.head(3).assign(kind="souveraen_falsch_von_probe_uebersehen")
    ex = pd.concat([caught, missed])
    ex["gold"] = ex["aliases"].apply(lambda a: a[0])
    cols = ["kind", "id"] + (["triviaqa_question_id"] if "triviaqa_question_id" in ex else []) + [
            "tier", "domain", "question", "gold", "answer", "verbal_conf",
            "mean_logprob", "probe_p_wrong", "probe_rank_pct"]
    ex[cols].to_csv(os.path.join(args.out, "examples.csv"), index=False)

    # ---------- Metriken ----------
    def at(f):
        return {k: float(escalation(y[te], S[k], [f], seed=args.seed)[0]) for k in S}
    by = df.assign(ok=df["correct"].astype(bool)).groupby(["tier", "domain"])["ok"].agg(["mean", "size"])
    metrics = {
        "model": meta["model"], "n_items": int(N), "accuracy": float(acc),
        "accuracy_by_tier_domain": {f"{a}/{b}": {"acc": float(r["mean"]), "n": int(r["size"])}
                                    for (a, b), r in by.iterrows()},
        "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                  "test_errors": int(y[te].sum()), "seed": args.seed,
                  "procedure": "stratified 60/20/20; layer+C chosen on val; refit train+val; test once"},
        "probe": {"best_layer": l_best, "n_layers": int(L - 1), "C": C_best,
                  "val_auroc_best": float(np.nanmax(val_grid)),
                  "val_auroc_by_layer": [None] + [float(x) for x in vb[1:]]},
        "test_auroc": auc, "test_auroc_ci95": ci, "paired_bootstrap": diffs,
        "escalation_errors_caught_test": {"at_10pct": at(0.1), "at_20pct": at(0.2), "at_30pct": at(0.3)},
        "robustness_repeated_splits": {"n_repeats": args.n_repeats, "test_auroc": rob_sum,
                                       "win_rate": win, "chosen_layers": rob_layers,
                                       "errors_caught_at_20pct_mean":
                                           {k: float(np.mean(v)) for k, v in rob_esc.items()}},
        "verbal_confidence": {"parse_rate": parse_rate,
                              "share_ge_90": float(np.mean(verbal_f >= 90)),
                              "n_distinct_values": int(len(np.unique(verbal_f)))},
        "notes": ["Target: y=1 means the model's answer is WRONG; AUROC = error detection.",
                  "Baselines are untrained scores, evaluated on the identical test items.",
                  "Ties in rankings broken at random (escalation), not in favour of any method.",
                  "Results are specific to this model, prompt and question distribution."],
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[analyze] geschrieben nach {args.out}/: roc.png escalation.png metrics.json examples.csv")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--phase", choices=["all", "collect", "analyze"], default="all")
    ap.add_argument("--dataset", choices=["inline", "triviaqa"], default="inline")
    ap.add_argument("--limit", type=int, default=0, help="0 = alle Fragen")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--dtype", default="auto", help="auto|float32|float16|bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    if args.phase in ("all", "collect"):
        collect(args)
    if args.phase in ("all", "analyze"):
        analyze(args)


if __name__ == "__main__":
    main()
