"""
=============================================================
CONFIDENCE-WEIGHTED SENTIMENT ANALYSIS
=============================================================

PROBLEM STATEMENT
-----------------
Existing sentiment analysis tools treat all input text as equally
reliable, ignoring the effect of text length and lexical diversity
on prediction confidence. Short or repetitive text (e.g., "bad bad bad")
provides less signal than a rich, multi-faceted review. This project
proposes a confidence-weighted sentiment classifier that adjusts its
certainty score based on textual richness metrics — specifically,
token count and type-token ratio (TTR) — producing more calibrated
and interpretable predictions.

APPROACH
--------
1. Baseline sentiment scoring using VADER (rule-based, no training needed)
2. Textual richness computation: token count + type-token ratio (TTR)
3. Confidence weighting: scale raw VADER compound score by a richness
   factor derived from TTR and token count
4. Evaluation on a small labeled dataset with accuracy + confusion matrix

DEPENDENCIES: pip install vaderSentiment scikit-learn matplotlib
=============================================================
"""

import re
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ── Dataset ──────────────────────────────────────────────────────────────────

DATASET = [
    # (text, true_label)  labels: positive / negative / neutral
    ("This product is absolutely fantastic. Best purchase I've made all year. Build quality, speed, battery — all excellent.", "positive"),
    ("Terrible. Broke after two days. Customer support ghosted me. Complete waste of money.", "negative"),
    ("It's okay. Does what it says. Nothing special, nothing broken.", "neutral"),
    ("I love it love it love it love it love it so much.", "positive"),
    ("Bad bad bad. Worst ever.", "negative"),
    ("The interface is clean and responsive. Setup took under five minutes. Impressed by the attention to detail.", "positive"),
    ("Arrived late, packaging was crushed, and the item itself smells strange. Returning immediately.", "negative"),
    ("Product received. Used once.", "neutral"),
    ("Mixed feelings. Hardware is gorgeous — best build I've held in years. Software, though, is a buggy disaster.", "neutral"),
    ("Exceeded expectations on every front. Highly recommend to anyone in the market.", "positive"),
    ("Not what I expected. Description was misleading. Feels cheap.", "negative"),
    ("Does the job. Average quality for the price.", "neutral"),
    ("Outstanding quality. Superb finish. Will definitely buy again without hesitation.", "positive"),
    ("Stopped working after a week. Replacement also failed. Avoid.", "negative"),
    ("Fine. Neutral. Neither good nor bad.", "neutral"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())

def type_token_ratio(tokens: list[str]) -> float:
    """Lexical diversity: unique tokens / total tokens. Ranges 0–1."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def richness_weight(tokens: list[str]) -> float:
    """
    Combine TTR and log-scaled token count into a [0, 1] weight.
    Short, repetitive text → low weight → lower confidence.
    """
    ttr = type_token_ratio(tokens)
    # log normalise token count; saturates around 50 tokens
    length_factor = min(1.0, math.log1p(len(tokens)) / math.log1p(50))
    return (ttr + length_factor) / 2  # simple average of both factors

def compound_to_label(score: float, threshold: float = 0.05) -> str:
    if score >= threshold:
        return "positive"
    elif score <= -threshold:
        return "negative"
    return "neutral"

# ── Classifier ────────────────────────────────────────────────────────────────

analyzer = SentimentIntensityAnalyzer()

def classify(text: str) -> dict:
    tokens      = tokenize(text)
    ttr         = type_token_ratio(tokens)
    weight      = richness_weight(tokens)
    scores      = analyzer.polarity_scores(text)
    raw_compound = scores["compound"]

    # Scale compound toward 0 when text is poor quality
    weighted_compound = raw_compound * weight

    return {
        "text":             text,
        "token_count":      len(tokens),
        "ttr":              round(ttr, 3),
        "richness_weight":  round(weight, 3),
        "raw_compound":     round(raw_compound, 3),
        "weighted_compound":round(weighted_compound, 3),
        "raw_label":        compound_to_label(raw_compound),
        "weighted_label":   compound_to_label(weighted_compound),
    }

# ── Run ───────────────────────────────────────────────────────────────────────

results   = [classify(text) for text, _ in DATASET]
true_labels = [label for _, label in DATASET]

raw_preds      = [r["raw_label"]      for r in results]
weighted_preds = [r["weighted_label"] for r in results]

raw_acc      = accuracy_score(true_labels, raw_preds)
weighted_acc = accuracy_score(true_labels, weighted_preds)

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Text':<55} {'TTR':>5} {'Wt':>5} {'Raw':>9} {'Wtd':>9} {'True':>9}")
print("-" * 95)
for r, (_, truth) in zip(results, DATASET):
    snippet = r["text"][:52] + "..." if len(r["text"]) > 52 else r["text"]
    print(f"{snippet:<55} {r['ttr']:>5.2f} {r['richness_weight']:>5.2f} "
          f"{r['raw_label']:>9} {r['weighted_label']:>9} {truth:>9}")

print(f"\nBaseline VADER accuracy : {raw_acc:.1%}")
print(f"Confidence-weighted acc : {weighted_acc:.1%}")

# ── Visualisations ────────────────────────────────────────────────────────────

LABEL_ORDER = ["positive", "neutral", "negative"]
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Confidence-Weighted Sentiment Analysis", fontsize=15, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# 1. Confusion matrix — baseline
ax1 = fig.add_subplot(gs[0, 0])
cm1 = confusion_matrix(true_labels, raw_preds, labels=LABEL_ORDER)
ConfusionMatrixDisplay(cm1, display_labels=LABEL_ORDER).plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title(f"Baseline VADER\naccuracy = {raw_acc:.1%}", fontsize=11)

# 2. Confusion matrix — weighted
ax2 = fig.add_subplot(gs[0, 1])
cm2 = confusion_matrix(true_labels, weighted_preds, labels=LABEL_ORDER)
ConfusionMatrixDisplay(cm2, display_labels=LABEL_ORDER).plot(ax=ax2, colorbar=False, cmap="Greens")
ax2.set_title(f"Confidence-Weighted\naccuracy = {weighted_acc:.1%}", fontsize=11)

# 3. Richness weight vs |raw compound| scatter
ax3 = fig.add_subplot(gs[1, 0])
weights    = [r["richness_weight"]       for r in results]
raw_abs    = [abs(r["raw_compound"])     for r in results]
wtd_abs    = [abs(r["weighted_compound"])for r in results]
colors     = ["#1D9E75" if t == "positive" else "#D85A30" if t == "negative" else "#888780"
              for _, t in DATASET]
ax3.scatter(weights, raw_abs, c=colors, alpha=0.7, edgecolors="white", linewidth=0.8,
            label="raw |compound|", marker="o", s=80)
ax3.scatter(weights, wtd_abs, c=colors, alpha=0.4, edgecolors="none",
            label="weighted |compound|", marker="^", s=70)
ax3.set_xlabel("Richness weight")
ax3.set_ylabel("|Compound score|")
ax3.set_title("Richness weight vs sentiment strength")
ax3.legend(fontsize=9)
ax3.grid(True, linewidth=0.4, alpha=0.5)

# 4. TTR bar chart per sample
ax4 = fig.add_subplot(gs[1, 1])
indices = range(len(results))
bar_colors = ["#1D9E75" if t == "positive" else "#D85A30" if t == "negative" else "#888780"
              for _, t in DATASET]
ax4.bar(indices, [r["ttr"] for r in results], color=bar_colors, edgecolor="white", linewidth=0.6)
ax4.set_xlabel("Sample index")
ax4.set_ylabel("Type-token ratio (TTR)")
ax4.set_title("Lexical diversity per sample")
ax4.set_xticks(list(indices))
ax4.set_xticklabels([str(i) for i in indices], fontsize=8)
ax4.grid(axis="y", linewidth=0.4, alpha=0.5)

# Legend patch
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1D9E75", label="positive"),
    Patch(facecolor="#D85A30", label="negative"),
    Patch(facecolor="#888780", label="neutral"),
]
ax4.legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.savefig("sentiment_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to sentiment_results.png")
