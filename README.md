\# Confidence-Weighted Sentiment Analysis



A sentiment classifier that adjusts its confidence based on how informative the input text is.



\## Problem



Standard sentiment tools are equally confident whether the input is a single repeated word or a detailed paragraph. This project adds a weighting layer that lowers confidence for short or repetitive text.



\## How It Works



1\. Runs sentiment scoring using VADER

2\. Computes two richness metrics per input — token count and type-token ratio (TTR)

3\. Scales the confidence score down for low-richness text

4\. Compares weighted vs baseline accuracy on a labeled dataset



\## Setup



```bash

pip install vaderSentiment scikit-learn matplotlib

```



\## Usage



```bash

python sentiment\_analysis.py

```



Outputs a results table in the terminal and saves a `sentiment\_results.png` with four plots — confusion matrices, a richness vs sentiment scatter, and a per-sample TTR bar chart.



\## Results



| Model | Accuracy |

|---|---|

| Baseline VADER | see terminal output |

| Confidence-Weighted | see terminal output |



\## Dependencies



\- `vaderSentiment` — sentiment scoring

\- `scikit-learn` — evaluation metrics

\- `matplotlib` — visualizations

