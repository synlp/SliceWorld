import math
import re
from collections import Counter
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


FOCAL_LESION_TERMS = [
    "mass",
    "nodule",
    "lesion",
    "tumor",
    "tumour",
    "metastasis",
    "metastases",
    "focal opacity",
    "focal abnormality",
]


def feature_summary(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(prediction, target).item()
    cosine = F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=-1).mean().item()
    return {"mse": mse, "cosine": cosine}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(predictions: Iterable[str], references: Iterable[str], max_n: int = 4) -> Dict[str, float]:
    predictions = list(predictions)
    references = list(references)
    result = {}
    pred_lengths = 0
    ref_lengths = 0
    precisions = []
    for n in range(1, max_n + 1):
        overlap = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = _tokens(pred)
            ref_tokens = _tokens(ref)
            pred_lengths += len(pred_tokens) if n == 1 else 0
            ref_lengths += len(ref_tokens) if n == 1 else 0
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts = _ngram_counts(ref_tokens, n)
            total += sum(pred_counts.values())
            overlap += sum((pred_counts & ref_counts).values())
        precision = (overlap + 1.0) / (total + 1.0)
        precisions.append(precision)
        result[f"bleu_{n}"] = precision
    brevity = 1.0 if pred_lengths > ref_lengths else math.exp(1.0 - ref_lengths / max(pred_lengths, 1))
    result["bleu"] = brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return result


def rouge_l(predictions: Iterable[str], references: Iterable[str]) -> Dict[str, float]:
    scores = []
    for pred, ref in zip(predictions, references):
        a = _tokens(pred)
        b = _tokens(ref)
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, token_a in enumerate(a, start=1):
            for j, token_b in enumerate(b, start=1):
                if token_a == token_b:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        lcs = table[-1][-1]
        precision = lcs / max(len(a), 1)
        recall = lcs / max(len(b), 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-8))
    return {"rouge_l": sum(scores) / max(len(scores), 1)}


def text_metrics(predictions: Iterable[str], references: Iterable[str]) -> Dict[str, float]:
    predictions = list(predictions)
    references = list(references)
    output = corpus_bleu(predictions, references)
    output.update(rouge_l(predictions, references))
    return output


def contains_focal_lesion_mention(text: str) -> bool:
    lowered = text.lower()
    for term in FOCAL_LESION_TERMS:
        if term in lowered and not re.search(rf"\b(no|without|absent|negative for)\b[^.]*\b{re.escape(term)}\b", lowered):
            return True
    return False


def counterfactual_lesion_summary(factual: List[str], lesion_zero: List[str]) -> Dict[str, float]:
    before = [contains_focal_lesion_mention(text) for text in factual]
    after = [contains_focal_lesion_mention(text) for text in lesion_zero]
    positives = [i for i, value in enumerate(before) if value]
    negatives = [i for i, value in enumerate(before) if not value]
    removal = sum(before[i] and not after[i] for i in positives) / max(len(positives), 1)
    hallucination = sum(after[i] for i in negatives) / max(len(negatives), 1)
    return {"target_removal_rate": removal, "negative_hallucination_rate": hallucination}
