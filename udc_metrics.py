import functools
import tensorflow as tf


def create_evaluation_metrics():
    eval_metrics = {}
    N = 10
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = tf.contrib.learn.MetricSpec(
            metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k,
            k=k))
        eval_metrics["ap_at_%d" % k] = tf.contrib.learn.MetricSpec(
            metric_fn=functools.partial(
                tf.metrics.sparse_average_precision_at_k,
                k=k))
    eval_metrics["MAP"] = tf.contrib.learn.MetricSpec(
        metric_fn=functools.partial(
            tf.metrics.sparse_average_precision_at_k,
            k=N))
    # eval_metrics["MRR"] = MetricSpec(metric_fn=evaluate_mrr)
    return eval_metrics

def evaluate_mrr(labels, predictions):
    MRR = 0
    pairs_all = list(zip(predictions, labels))
    for preds, true_label in pairs_all:
        rank1 = next(filter(lambda x: x[1][0] == true_label,
                            enumerate(
                                sorted(
                                    enumerate(preds),
                                    key=lambda x: x[1],
                                    reverse=True))))[0] + 1
        MRR += 1 / rank1
    MRR /= len(pairs_all)
    print("MRR:", MRR)
    return MRR

def evaluate_map(labels, predictions):
    MAP = 0
    print(labels)
    print(predictions)
    pairs_all = list(zip(predictions, labels))
    for probs, true_label in pairs_all:
        ohe = np.zeros(len(probs))
        ohe[true_label] = 1
        pairs = zip(ohe, probs)
        p, AP = 0, 0
        for k, (label, prob) in enumerate(sorted(pairs,
                                                 key=lambda x: x[-1],
                                                 reverse=True)):
            if label == 1:  # top-all
                p += 1
                AP += p / (k + 1)
        assert(p != 0)
        AP /= p  # n_label1s
        MAP += AP
    MAP /= len(pairs_all)
    print("MAP:", MAP)
    return MAP
