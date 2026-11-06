# Sleep Apnea Detection

Learning and portfolio project: detect sleep apnea from UCDDB overnight polysomnography recordings using multi-channel signal processing, hand-crafted feature extraction, and classical machine learning (XGBoost).

## Language

**Recording**:
One participant's overnight polysomnography study in UCDDB (`<id>.rec` signal file plus `<id>_respevt.txt` and `<id>_stage.txt` annotations).
_Avoid_: Sample, night, study

**Window**:
A fixed-length time segment cut from a Recording; the unit from which features are extracted and classified as apnea or normal.
_Avoid_: Epoch, segment, chunk

**Apnea event**:
An annotated respiratory interval (`APNEA-O/C/M` or `HYP-O/C/M`) in a Recording where breathing stops or is reduced; the source of Window labels.
_Avoid_: Episode, incident

**Sleep stage**:
The annotated physiological state (`0`=Wake, `1`=REM, `2`–`5`=NREM, `8`=Indeterminate) assigned to each 30-second interval of a Recording.
_Avoid_: Phase, state

**Feature vector**:
The tabular row of time-domain, frequency-domain, and physiological summary statistics extracted from a single Window's filtered signals.
_Avoid_: Embedding, representation

**AHI**:
Apnea-Hypopnea Index: Apnea events per hour of sleep in a Recording; derived from Window predictions, not predicted directly.
_Avoid_: Severity score
