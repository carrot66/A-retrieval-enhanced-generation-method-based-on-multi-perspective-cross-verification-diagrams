# Reproducibility

The paper-facing release evaluates FaithEval, RealtimeQA, MuSiQue-Golden,
MuSiQue-Negative, SQuAD-Golden, and SQuAD-Negative. Llama runs all six files;
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 run the four conflict files.
The reported experiments are zero-shot with temperature 0 and workers=12.

The label-separation utility in code/data_preparation/isolate_labels.py creates
an inference file without answer fields and a separate answer-key file. The
source runner receives only the inference file during decision making. Gold
answers are joined after the final option ID has been fixed.

code/scripts/replay_formal.sh reads the retained source and semantic traces and
rebuilds per-sample projections. Dataset-level accuracy follows the official
TruthfulRAG evaluator in code/source_generation/evaluate.py. The committed
PAPER_REPORT files contain the paper-facing aggregate results.