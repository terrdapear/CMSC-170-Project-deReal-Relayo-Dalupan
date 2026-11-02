from time import perf_counter

# Republic Act is the positive class
POSITIVE = 1


def get_confusion(actual, predicted):
    '''
    returns
    tp, fn, fp, tn
    '''
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(actual)):
        if actual[i] == POSITIVE:
            if predicted[i] == POSITIVE:
                tp += 1
            else:
                fn += 1
        else:
            if predicted[i] == POSITIVE:
                fp += 1
            else:
                tn += 1
    return tp, fn, fp, tn


def safe_divide(numerator, denominator):
    # ex. precision has nothing to divide by if nothing was predicted positive
    if denominator == 0:
        return 0
    return numerator / denominator


def accuracy(tp, fn, fp, tn):
    # (TP + TN) / (TP + TN + FP + FN)
    return safe_divide(tp + tn, tp + tn + fp + fn)


def precision(tp, fp):
    # TP / (TP + FP)
    return safe_divide(tp, tp + fp)


def recall(tp, fn):
    # TP / (TP + FN)
    return safe_divide(tp, tp + fn)


def f1_score(p, r):
    # 2 * precision * recall / (precision + recall)
    return safe_divide(2 * p * r, p + r)


def evaluate(model, paragraphs, labels):
    # any model works as long as it has classify_paragraph
    start = perf_counter()
    predicted = []
    for paragraph in paragraphs:
        predicted.append(model.classify_paragraph(paragraph))
    runtime = perf_counter() - start

    tp, fn, fp, tn = get_confusion(labels, predicted)
    p = precision(tp, fp)
    r = recall(tp, fn)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy(tp, fn, fp, tn),
        "precision": p,
        "recall": r,
        "f1": f1_score(p, r),
        "runtime": runtime,
    }
