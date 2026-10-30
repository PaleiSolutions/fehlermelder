"""Fügt den veröffentlichten TriviaQA-Ergebnisdateien Fragetext und Aliasliste wieder hinzu.

Die Fragen von TriviaQA werden nicht mit diesem Repository verteilt (ungeklärte Lizenz).
Dieses Skript lädt den Datensatz von HuggingFace und verknüpft über triviaqa_question_id.

Aufruf:  python join_triviaqa.py results/qwen2.5-3b_triviaqa/generations.csv  > mit_fragen.csv
"""
import sys

import pandas as pd
from datasets import load_dataset

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
lookup = {r["question_id"]: (r["question"], " ; ".join([r["answer"]["value"]] + r["answer"]["aliases"]))
          for r in ds}
df = pd.read_csv(sys.argv[1])
df["question"] = df["triviaqa_question_id"].map(lambda q: lookup[q][0])
df["aliases"] = df["triviaqa_question_id"].map(lambda q: lookup[q][1])
df.to_csv(sys.stdout, index=False)
