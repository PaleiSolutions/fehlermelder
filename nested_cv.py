# Aufruf: python nested_cv.py outputs_3b/cache_Qwen__Qwen2.5-3B-Instruct.npz outputs_3b/nested_cv.json
"""Post-hoc (nach dem Haupt-Split ergänzt): verschachtelte 5-fach-CV. Jede Frage ist genau einmal Testfall.
Innen: Schicht + C per stratifiziertem 75/25-Holdout im äußeren Trainingsteil (wie Hauptprotokoll),
dann Refit auf dem ganzen äußeren Trainingsteil. Baselines untrainiert auf denselben Items.
Maßgeblich ist auroc_fold_mean: Gepoolte Out-of-Fold-AUROC benachteiligt die Probe, weil
jedes Fold-Modell (andere Schicht/C) anders skalierte Wahrscheinlichkeiten liefert."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script import split, select_and_test
z = np.load(sys.argv[1]); H = z["H"].astype(np.float32); meta = json.loads(str(z["meta"]))
df = pd.DataFrame(meta["rows"]); y = (~df["correct"].astype(bool)).to_numpy().astype(int)
base = {"logprob_mean": -df["mean_logprob"].to_numpy(), "logprob_min": -df["min_logprob"].to_numpy(),
        "verbal": -df["verbal_conf"].fillna(df["verbal_conf"].median()).to_numpy()}
res = []
for rep in range(3):
    oof = np.zeros(len(y)); layers = []; fold_auc = []
    for k, (tr_o, te_o) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(H[:, 0], y)):
        # innerer Split: 75/25 von tr_o, Test = te_o (wird von select_and_test nur einmal benutzt)
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(tr_o, train_size=0.75, stratify=y[tr_o], random_state=rep * 10 + k)
        lb, C, _, p = select_and_test(H, y, tr, va, te_o, rep * 10 + k)
        oof[te_o] = p; layers.append(lb)
        # AUROC je Fold (keine Vermischung unterschiedlich skalierter Fold-Modelle)
        fold_auc.append({"probe": roc_auc_score(y[te_o], p),
                         **{m: roc_auc_score(y[te_o], b[te_o]) for m, b in base.items()}})
        print(f"rep {rep} fold {k}: layer {lb}, C {C}", flush=True)
    rng = np.random.default_rng(rep); S = {"probe": oof, **base}
    auc = {m: roc_auc_score(y, s) for m, s in S.items()}
    d = {m: [] for m in base}
    for _ in range(2000):
        b = rng.integers(0, len(y), len(y))
        a = roc_auc_score(y[b], oof[b])
        for m in base: d[m].append(a - roc_auc_score(y[b], base[m][b]))
    fa = pd.DataFrame(fold_auc)
    out = {"rep": rep, "layers": layers, "auroc_pooled": auc,
           "auroc_fold_mean": {m: float(fa[m].mean()) for m in fa},
           "fold_diff_probe_minus": {m: [float(x) for x in (fa["probe"] - fa[m])] for m in base},
           "diff": {m: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                        "p_le_0": float(np.mean(np.array(v) <= 0))} for m, v in d.items()}}
    print(json.dumps(out), flush=True); res.append(out)
json.dump(res, open(sys.argv[2], "w"), indent=1)
# Zusammenfassung über alle Wiederholungen x Folds (Folds überlappen nicht innerhalb einer Wiederholung)
for m in base:
    d = np.concatenate([r["fold_diff_probe_minus"][m] for r in res])
    print(f"fold-gemittelt probe - {m}: {d.mean():+.3f}  (Folds mit Vorsprung: {(d > 0).sum()}/{len(d)})")
