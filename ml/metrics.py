import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from ml.calibration import calibration


def evaluate(y, p):
    y, p = np.asarray(y), np.asarray(p)
    predictions = p >= .5
    details = classification_report(y, predictions, labels=[0, 1],
                                    target_names=["Real", "Fake"], output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y, predictions), "macro_f1": details["macro avg"]["f1-score"],
            "per_class": {name: details[name] for name in ("Real", "Fake")},
            "roc_auc": roc_auc_score(y, p), "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            "calibration": calibration(p, y)}
