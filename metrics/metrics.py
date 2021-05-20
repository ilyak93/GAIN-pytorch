import numpy as np

def calc_sensitivity(labels, predictions, false_alarm_points=None, return_thresholds_for_fa=False):
    if false_alarm_points is None:
        false_alarm_points = [.1 / 100, .05 / 100]

    fpr, tpr, auc, thresholds = roc_curve(labels, predictions)
    all_sens = []
    all_thresholds = []
    for fa in false_alarm_points:
        sensitivity_at_fa = np.interp(x=fa, xp=fpr, fp=tpr)
        th_at_fa = np.interp(x=fa, xp=fpr, fp=thresholds)
        all_sens += [sensitivity_at_fa]
        all_thresholds += [th_at_fa]

    if return_thresholds_for_fa:
        return all_sens, all_thresholds
    return all_sens, auc


def roc_curve(labels, preds, thresholds_count=10000):
    if len(labels) == 1:
        raise Exception(f'roc_curve: labels parameter is empty')
    if len(np.unique(labels)) == 1:
        raise Exception(f'roc_curve: labels parameter is composed of only one value')

    preds_on_positive = preds[labels == 1]
    preds_on_negative = preds[labels == 0]
    min_negative = min(preds_on_negative)
    max_positive = max(preds_on_positive)
    margin = 0  # (max_positive - min_negative)/100

    thresholds = np.linspace(min_negative - margin, max_positive + margin, thresholds_count)
    true_positive_rate = [np.mean(preds_on_positive > t) for t in thresholds]
    spec = [np.mean(preds_on_negative <= t) for t in thresholds]
    false_positive_rate = [1 - s for s in spec]
    auc = np.trapz(true_positive_rate, spec)

    thresholds = np.flip(thresholds, axis=0)
    false_positive_rate.reverse(), true_positive_rate.reverse()
    false_positive_rate, true_positive_rate = np.asarray(false_positive_rate), np.asarray(true_positive_rate)
    return false_positive_rate, true_positive_rate, auc, thresholds
