## 默认模型在测试时输出的顺序和测试集顺序相同
## The inputs must not be shuffled during evaluation.


import html
import re
from collections import Counter
from rouge import Rouge
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu
# NOTE: `load_metric` was removed from `datasets` in 3.0. It is only needed for
# SARI (20Minuten). Import it lazily inside caculate_sari so the rest of the
# metrics work on any `datasets` version (and with no `datasets` at all).

_ROUGE_L_SCORER = Rouge(metrics=["rouge-l"])


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(results)
    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    scores = _ROUGE_L_SCORER.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)
    avg_rouge = sum(rouges) / len(results)
    return avg_rouge


########################
## Accuracy (EM)
########################
def caculate_accuracy(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        if prediction == target:
            scores += 1
    avg_score = scores / len(results)
    return avg_score


########################
## F1-micro
########################
def f1_score(list1, list2):
    # TP: item in list1 and list2
    # FP: item in list1 but not in list2
    # TN: item not in list1 and list2
    # FN: item in list2 but not in list1
    num_TP = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                num_TP += 1
                break
    precision = num_TP / len(list1)
    recall = num_TP / len(list2)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall / (precision + recall))


def caculate_f1(results, data):
    scores = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if len(prediction) == 0 or len(target) == 0:
            continue
        score = f1_score(target, prediction)
        scores.append(score)
    avg_score = sum(scores) / len(results)
    return avg_score


########################
## fuzzywuzzy
########################
def caculate_fuzz(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        scores += fuzz.ratio(prediction, target)
    avg_score = scores / len(results) 
    return avg_score


########################
## SARI
########################
def caculate_sari(inputs, results, data):
    """Hugging Face/Tensor2Tensor SARI, implemented locally and offline.

    Returns the corpus mean on the conventional 0..100 scale. TRACE has one
    simplification reference per source.
    """
    if not (len(inputs) == len(results) == len(data)):
        raise ValueError("SARI sources, predictions, and references differ in length")
    if not results:
        return 0.0
    scores = [
        _sari_sentence(_sari_normalize(source), _sari_normalize(prediction),
                       [_sari_normalize(reference)])
        for source, prediction, reference in zip(inputs, results, data)
    ]
    return 100.0 * sum(scores) / len(scores)


def _sari_normalize(sentence):
    """Moses/sacreBLEU 13a tokenization used by HF's SARI metric."""
    sentence = html.unescape(sentence.lower())
    sentence = re.sub(r"<skipped>", "", sentence)
    sentence = re.sub(r"\n-", "", sentence)
    sentence = re.sub(r"\n", " ", sentence)
    sentence = f" {sentence} "
    sentence = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", sentence)
    sentence = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", sentence)
    sentence = re.sub(r"([\.,])([^0-9])", r" \1 \2", sentence)
    sentence = re.sub(r"([0-9])(-)", r"\1 \2 ", sentence)
    return " ".join(sentence.split())


def _sari_ngrams(tokens, order):
    return [" ".join(tokens[i:i + order])
            for i in range(max(0, len(tokens) - order + 1))]


def _sari_ngram(source, candidate, references, num_references):
    reference_counter = Counter(ngram for ref in references for ngram in ref)
    source_counter = Counter(source)
    source_repeated = Counter({key: value * num_references
                               for key, value in source_counter.items()})
    candidate_counter = Counter(candidate)
    candidate_repeated = Counter({key: value * num_references
                                  for key, value in candidate_counter.items()})

    keep = source_repeated & candidate_repeated
    keep_good = keep & reference_counter
    keep_all = source_repeated & reference_counter
    keep_precision = 1.0
    keep_recall = 1.0
    if keep:
        keep_precision = sum(keep_good[x] / keep[x] for x in keep_good) / len(keep)
    if keep_all:
        keep_recall = sum(keep_good.values()) / sum(keep_all.values())
    keep_score = (2 * keep_precision * keep_recall /
                  (keep_precision + keep_recall)
                  if keep_precision > 0 or keep_recall > 0 else 0.0)

    delete = source_repeated - candidate_repeated
    delete_good = delete - reference_counter
    delete_precision = 1.0
    if delete:
        delete_precision = sum(delete_good[x] / delete[x]
                               for x in delete_good) / len(delete)

    add = set(candidate_counter) - set(source_counter)
    add_good = add & set(reference_counter)
    add_all = set(reference_counter) - set(source_counter)
    add_precision = len(add_good) / len(add) if add else 1.0
    add_recall = len(add_good) / len(add_all) if add_all else 1.0
    add_score = (2 * add_precision * add_recall /
                 (add_precision + add_recall)
                 if add_precision > 0 or add_recall > 0 else 0.0)
    return keep_score, delete_precision, add_score


def _sari_sentence(source, candidate, references):
    source_tokens = source.split(" ")
    candidate_tokens = candidate.split(" ")
    reference_tokens = [reference.split(" ") for reference in references]
    keep_scores, delete_scores, add_scores = [], [], []
    for order in range(1, 5):
        keep, delete, add = _sari_ngram(
            _sari_ngrams(source_tokens, order),
            _sari_ngrams(candidate_tokens, order),
            [_sari_ngrams(tokens, order) for tokens in reference_tokens],
            len(references),
        )
        keep_scores.append(keep)
        delete_scores.append(delete)
        add_scores.append(add)
    return ((sum(keep_scores) / 4 + sum(delete_scores) / 4 +
             sum(add_scores) / 4) / 3)
