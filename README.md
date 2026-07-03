# fluency-from-alignment
### Predicting Spoken Fluency from Pause Structure with Forced Alignment + VAD

An end-to-end pipeline that predicts human fluency ratings of spoken English from the
timing and pause structure of the audio, using Qwen3 forced alignment and energy-based
voice activity detection, validated against expert scores.

---

## The Problem

When someone reads a passage aloud, how do we automatically judge how *fluently* they
read it — smoothly and without unnecessary pauses — the way a human rater would?

This is a core task in automated scoring of spoken assessments. The challenge is that
fluency lives in the *timing* of speech: where the pauses fall, how long they last, how
fast the words come. This project extracts that timing structure from audio and models
it against expert human fluency scores.

**The key insight:** the human fluency label in speechocean762 is defined as speaking
*smoothly and without unnecessary pauses*. That means pause structure — measured directly
from the audio — is a directly interpretable signal for the human judgment, without needing
to recognize *what* was said.

---

## Approach

1. **Forced alignment** (Qwen3-ForcedAligner-0.6B) → word-level timestamps, used where
   speech aligns cleanly (word counts, timing on fluent speech).
2. **Pause & timing features** from energy-based VAD: pause count, pause durations,
   silence ratio, articulation/speech rate, phonation ratio, onset latency, long-pause count.
3. **Validation** against 5-expert human fluency scores (Spearman); interpretable model +
   SHAP to identify which timing features drive the human rating.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Alignment vs ASR | Forced alignment (text known) | Read-aloud task — reference text is known, so skip ASR and its transcription errors |
| Pause measurement | Energy-based VAD, not alignment timestamps | Forced alignment breaks on disfluent speech; VAD is text-independent and stays robust |
| Fluency target | Continuous 0–10, Spearman validation | Labels are heavily skewed (~82% score 7–10); accuracy would reward a "predict high" baseline |
| Edge silence | Trimmed before rate/ratio features | Leading/trailing silence depends on when *record* was pressed — a recording artifact, not fluency |
| Word counts for rates | From known reference text, not the aligner | Text is known, so word count is reliable even where alignment fails |
| Aligner model | Qwen3-ForcedAligner-0.6B | Word-level timestamps, surpasses WhisperX/MFA in accuracy; runs on Apple Silicon |
| Pause threshold | 250 ms min (500 ms = "long pause") | Standard thresholds in the fluency literature |
| VAD threshold | Adaptive (per-utterance p10–p90) | Adapts to each recording's gain instead of a fixed global cutoff |

---

## Why not just use the alignment timestamps?

Forced alignment assumes the audio contains exactly the reference words, in order. This
holds for fluent readers but breaks on disfluent speech — precisely the low-fluency tail
we most care about.

We verified this empirically. On a disfluent utterance (human fluency = 3), the aligner
piled the reference words into the first half of the audio and left the entire second half
unmapped (reported as "trailing silence"). An energy plot revealed that region was not
silence at all — it was speech the aligner failed to place:

![Alignment fails on disfluent speech](figures/alignment_fails_on_disfluent.png)

The failure is systematic, not incidental. Measured across all 2,500 training utterances,
every alignment-failure signal — trailing "silence", zero-duration words, and utterance
duration — falls monotonically as human fluency rises:

![Alignment degrades as fluency drops](figures/alignment_degradation_by_fluency.png)

So we measure pauses directly from the audio using **energy-based Voice Activity Detection
(VAD)** — a technique that labels each short frame of audio as speech or silence based on
its energy (loudness), without needing to know what words were spoken. Concretely: we split
the audio into 25 ms frames, compute each frame's RMS energy, and threshold it (adaptively,
per utterance) into speech vs. silence. Silence stretches longer than 250 ms that fall
*between* speech are counted as pauses.

Because VAD looks only at the audio signal and not at any reference text, it stays reliable
on disfluent speech — it recovers the speech the aligner missed and detects the real pauses
(red) regardless of what was said:

![VAD recovers the speech the aligner missed](figures/vad_recovers_disfluent_speech.png)

---

## Notebooks

### Notebook 01 — Explore speechocean762
Load the dataset, confirm audio ships at 16 kHz, and analyze the fluency target
distribution. Finding: fluency is heavily right-skewed (~82% score 7–10, ~5% score ≤ 4),
which sets the modeling choice — continuous target validated with Spearman, with the
low-fluency tail as the region of interest.

### Notebook 02 — Forced alignment & the case for VAD
Run Qwen3 forced alignment and show it degrades systematically on disfluent speech (across
the full training set), motivating the switch to energy-based VAD for pause measurement.

---

## Data

[speechocean762](https://huggingface.co/datasets/mispeech/speechocean762) — 5,000 English
utterances from 250 non-native speakers (adults and children), Apache-2.0. Each utterance
scored by 5 experts at utterance / word / phoneme level (accuracy, fluency, completeness,
prosody). Audio ships at 16 kHz. Also carries speaker, gender, and age — enabling later
checks on whether fluency signals differ for children vs. adults (relevant to K-12).

**Note:** Data not tracked in git; downloaded via HuggingFace `datasets`.

---

## Setup

```bash
git clone https://github.com/KelyNorel/fluency-from-alignment.git
cd fluency-from-alignment
pyenv activate fluency-from-alignment
pip install -r requirements.txt

# The forced aligner is very new; if the model type isn't recognized,
# install transformers from source:
pip install "git+https://github.com/huggingface/transformers"

# speechocean762 downloads automatically via HuggingFace datasets on first run
```

---

## Stack

- **Qwen3-ForcedAligner-0.6B** — word-level forced alignment (via native Transformers support)
- **HuggingFace Transformers** — model loading and inference
- **HuggingFace datasets** — speechocean762 loading
- **PyTorch + MPS** — inference on Apple Silicon
- **NumPy** — energy-based Voice Activity Detection (VAD) and pause detection
- **scikit-learn** — interpretable fluency model
- **SHAP** — feature attribution for the human fluency rating
- **matplotlib** — visualization
- **Python 3.11**

---

## Limitations

- **Alignment on disfluent speech:** Forced alignment is unreliable in the low-fluency tail (documented above); pause features come from VAD instead, but word-level features that depend on alignment are less reliable there.
- **Short utterances:** speechocean762 utterances are a few seconds each, so per-utterance pause counts are low; signal comes from aggregating across 5,000 utterances.
- **Non-native, scripted speech:** The corpus is native-Mandarin speakers reading fixed prompts. Findings may not transfer directly to spontaneous speech or other L1 backgrounds.
- **Label skew:** Very few low-fluency examples (~5% score ≤ 4), which limits how finely the model can learn the disfluent end of the scale.

---


## Project Structure
```
fluency-from-alignment/

├── notebooks/

│   ├── 01_explore_data.ipynb         # dataset exploration + fluency target distribution

│   └── 02_forced_alignment.ipynb     # forced alignment + the case for VAD

├── figures/                          # committed plots used in this README

│   ├── alignment_fails_on_disfluent.png

│   └── vad_recovers_disfluent_speech.png

├── data/                             # not tracked in git (dataset cache, alignment parquet)

├── requirements.txt

└── README.md
```
---

**Author:** Raquel (Kely) Norel, PhD  
**Domain:** Speech Processing / Educational Measurement / NLP  
**Status:** 🚧 In progress — building pause/timing features from VAD
