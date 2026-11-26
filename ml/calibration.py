import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def fit_temperature(margins, labels):
    def nll(log_temperature):
        scaled = margins / np.exp(log_temperature)
        return np.mean(np.logaddexp(0, scaled) - labels * scaled)
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def calibration(probabilities, labels):
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (probabilities >= .5) == labels
    indices = np.minimum((confidence * 15).astype(int), 14)
    bins = []
    for index in range(15):
        mask = indices == index
        if mask.any():
            bins.append({"bin": index, "count": int(mask.sum()),
                         "confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
    ece = sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / len(labels)
    return {"ece": float(ece), "mean_confidence": float(confidence.mean()), "bins": bins}


def probability(margin, temperature):
    return expit(np.asarray(margin) / temperature)
