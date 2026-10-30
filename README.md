# Fehlermelder

Sagen die inneren Aktivierungen eines offenen Sprachmodells vorher, ob seine gerade gegebene Antwort **falsch** ist – und zwar besser als (a) die Selbstauskunft des Modells („Wie sicher bist du, 0–100?“) und (b) die Wahrscheinlichkeit seiner eigenen Wörter?

Kleines Experiment mit zwei kleinen Modellen (Qwen2.5-1.5B und -3B-Instruct) zum Essay **[„Die KI ahnt ihre Fehler – sie sagt es nur nicht“](ESSAY-URL)**. Begleitmaterial zum Text, kein Benchmark, kein Produktionscode, nicht gewartet.

Versuchsdesign und Prüfkriterien: Alexander Hertsch. Implementierung und Auswertung: KI-gestützt.

*English summary: A small, honest experiment accompanying a German-language essay. A logistic-regression probe on the hidden states of Qwen2.5-3B-Instruct detects wrong short factual answers better than the model's verbalized confidence (AUROC ≈ 0.88–0.92 vs. ≈ 0.62–0.64) and modestly but consistently better than mean token log-probability (≈ +0.04 AUROC). For Qwen2.5-1.5B no reliable advantage over log-probability was found. Results are specific to these models, prompts and question sets.*

## Ergebnisse

Maßgeblich ist die verschachtelte 5-fach-Kreuzvalidierung (3 Wiederholungen): Jede Frage ist genau einmal Testfall, Schicht und Regularisierung werden in jedem Fold nur auf dessen Trainingsteil gewählt. Angegeben ist der AUROC für die Erkennung falscher Antworten, gemittelt über die 15 Folds.

| | Genauigkeit | Probe | Mittl. Log-Prob | Min. Log-Prob | Verbale Sicherheit | Probe vor mittl. Log-Prob |
|---|---|---|---|---|---|---|
| 3B, TriviaQA (1000) | 44,6 % | **0,879** | 0,833 | 0,834 | 0,644 | +0,046 (14/15 Folds) |
| 3B, eigene Fragen (399) | 86,0 % | **0,922** | 0,883 | 0,887 | 0,615 | +0,039 (13/15 Folds) |
| 1.5B, TriviaQA (400) | 36,5 % | 0,822 | 0,800 | 0,799 | 0,514 | +0,022 (11/15 Folds) |
| 1.5B, eigene Fragen (399) | 78,7 % | 0,903 | 0,900 | 0,908 | 0,545 | +0,003 (8/15 Folds) |

**Eskalation (3B, eigene Fragen, Fehlerrate 14 %, Mittel über 20 Zufalls-Splits):** Werden die unsichersten 20 % der Antworten an einen Menschen gegeben, fängt man mit der Probe 79 % aller Fehler ab, mit dem mittleren Log-Prob 72 %, mit der verbalen Sicherheit 39 %.

**Selbstauskunft:** Das 3B-Modell nennt bei 82 % (TriviaQA) bzw. 96 % (eigene Fragen) aller Antworten eine Sicherheit ≥ 90. Das 1.5B-Modell antwortet ausnahmslos mit 95 oder 100.

**Schichtwahl:** Das Signal liegt bei beiden Modellen grob bei zwei Dritteln der Netztiefe (3B: meist Schicht 21–27 von 36).

### Wie die Zahlen zustande kommen – und warum es drei Auswertungen gibt

1. **Haupt-Split (vorab festgelegt):** stratifiziert 60/20/20, Seed 0, Schicht und C nur auf Validierung gewählt, Test genau einmal. Ergebnis in `metrics.json → test_auroc`. Beim 3B-Modell auf TriviaQA ergab dieser eine Split ein Patt zwischen Probe (0,840) und Log-Prob (0,843).
2. **20 weitere Zufalls-Splits**, jeweils mit eigener Schichtwahl (`metrics.json → robustness_repeated_splits`). Dort liegt die Probe beim 3B-Modell in 17–19 von 20 Splits vorn. Die Test-Mengen überlappen sich; das ist ein Stabilitätsbefund, kein Signifikanztest.
3. **Verschachtelte Kreuzvalidierung** (`nested_cv.py`, `nested_cv.json`), **nachträglich ergänzt**, nachdem der Haupt-Split unentschieden war. Sie nutzt alle Fragen als Testfälle. Maßgeblich ist `auroc_fold_mean`: Die gepoolte Out-of-Fold-AUROC benachteiligt die Probe, weil jedes Fold-Modell (andere Schicht, andere Regularisierung) anders skalierte Wahrscheinlichkeiten liefert. Die Baselines haben eine feste Skala und sind davon nicht betroffen.

## Starten

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt datasets

# eigener Fragenkatalog (questions.py)
python script.py --model Qwen/Qwen2.5-3B-Instruct --out results/qwen2.5-3b_inline
# TriviaQA, wird zur Laufzeit von HuggingFace geladen
python script.py --model Qwen/Qwen2.5-3B-Instruct --dataset triviaqa --limit 1000 --out results/qwen2.5-3b_triviaqa
# verschachtelte Kreuzvalidierung auf dem dabei entstandenen Cache
python nested_cv.py results/qwen2.5-3b_triviaqa/cache_Qwen__Qwen2.5-3B-Instruct.npz results/qwen2.5-3b_triviaqa/nested_cv.json
```

Gerät und Datentyp werden automatisch gewählt (CUDA → bf16, Apple Silicon/MPS → fp16, CPU → fp32). Die 3B-Läufe entstanden auf einem Mac mit 24 GB (MPS, fp16), die 1.5B-Läufe auf CPU (bf16). Auf anderer Hardware können einzelne generierte Antworten abweichen, weil sich die Numerik unterscheidet. Die Auswertung aus einem vorhandenen Cache (`--phase analyze`, `nested_cv.py`) ist dagegen deterministisch. Die Hidden-State-Caches (`*.npz`, bis 140 MB) sind wegen der Größe nicht im Repository.

## Was das Skript tut

1. **Antwort:** greedy, Kurzantwort-Prompt, gekürzt am ersten Zeilenumbruch.
2. **Label:** normalisierter Abgleich gegen eine Aliasliste (Kleinschreibung, Akzente, Interpunktion, Artikel, Plural-s). Korrekt ist eine Antwort, wenn ein Alias als zusammenhängende Wortfolge in ihr vorkommt.
3. **Merkmale:** Ein zweiter Durchlauf über Prompt plus exakt die generierten Antwort-Token liefert den Hidden State am letzten Antwort-Token (alle Schichten) und die Log-Probs der Antwort-Token.
4. **Selbstauskunft:** separater Aufruf in derselben Konversation: „How confident are you that your answer is correct? 0–100“.
5. **Probe:** Standardisierung plus L2-regularisierte logistische Regression mit ausgeglichenen Klassengewichten. Die Baselines werden nicht trainiert.
6. **Eskalationskurve:** Welcher Anteil aller Fehler wird abgefangen, wenn die x % unsichersten Antworten an einen Menschen gehen? Gleichstände (verbal: fast nur 95/100) werden zufällig aufgelöst, nicht zugunsten einer Methode.

## Daten

- **Eigene Fragen** (`questions.py`): 399 kurze Faktenfragen in drei Schwierigkeitsstufen, davon 107 aus Finanz und Banking (Basel III, DORA, Zahlungsverkehr, Bankenkrisen). Ausschließlich öffentliches Allgemeinwissen.
- **TriviaQA** (rc.nocontext, Validierung, 1000 bzw. 400 Fragen, Seed 0) wird nicht mitverteilt, weil die Lizenz ungeklärt ist. In den Ergebnisdateien stehen statt der Fragetexte die TriviaQA-IDs. `python join_triviaqa.py results/qwen2.5-3b_triviaqa/generations.csv > mit_fragen.csv` fügt Frage und Aliasliste lokal wieder hinzu.
- **Label-Prüfung:** Alle Falsch-Labels der eigenen Fragen und Stichproben der TriviaQA-Labels wurden von Hand geprüft. Bekannte Restfehler:
  - „Wb“ (Einheitenzeichen für Weber) zählt als falsch.
  - Grenzfälle: „Thimpu“ statt Thimphu, „O“ statt O negativ.
  - TriviaQA selbst ist verrauscht, etwa „Aether“ statt „Ether“ oder ein reiner Nachname, wenn nur der volle Name als Alias geführt wird.

## Grenzen

- **Modell- und aufgabenspezifisch:** Eine Probe gilt für das Modell, den Prompt und die Art Fragen, auf denen sie trainiert wurde. Sie überträgt sich erfahrungsgemäß schlecht auf andere Aufgaben.
- **Kleine Modelle, kurze Faktenantworten:** keine langen Texte, keine Antworten auf Basis von Dokumenten (RAG).
- **Braucht Labels:** einige hundert geprüfte Beispiele aus der Zielaufgabe.
- **Kleine Testmengen:** Einzelne Splits schwanken stark (siehe Haupt-Split oben); belastbar sind nur die Mittel über viele Splits bzw. Folds.
- **Kein Mechanismus:** Gezeigt ist nur, dass die Information linear auslesbar ist.

## Dateien

```
script.py          Pipeline: Antworten erzeugen, labeln, Merkmale abgreifen (collect) + Auswertung (analyze)
nested_cv.py       verschachtelte Kreuzvalidierung auf einem Cache
questions.py       eigener Fragenkatalog
join_triviaqa.py   TriviaQA-Fragetexte lokal wieder an Ergebnisdateien anfügen
results/<lauf>/    metrics.json, roc.png, escalation.png, examples.csv, generations.csv, nested_cv.json
```

Die 1.5B-Läufe entstanden mit einer Vorversion von `script.py`, die sich nur im Ausgabeformat unterscheidet (TriviaQA-ID-Spalte); die Auswertung wurde mit der vorliegenden Version gerechnet.

## Lizenz

MIT, siehe `LICENSE`.
