"""Interpret Ultralytics cut detector outputs and expose the winning label."""


def _names_dict(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _positive_ids(names):
    labels = _names_dict(names)
    positive = []
    for class_id, label in labels.items():
        normalized = "".join(ch for ch in label.lower() if ch.isalnum())
        if any(token in normalized for token in ("notcut", "uncut", "intact", "unseparated")):
            continue
        if "cut" in normalized or "separated" in normalized or "split" in normalized:
            positive.append(class_id)
    return labels, positive


def evaluate_cut_model_prediction(results, model=None, cut_class_id=0, min_conf=0.25):
    """Read either a classification result or a detection/segmentation result.

    If class names identify the positive class, use those labels. If names are
    generic, fall back to the configured class ID (class 0 by default).
    """
    result = results[0] if isinstance(results, (list, tuple)) and results else results
    names = getattr(model, "names", None) if model is not None else None
    if names is None and result is not None:
        names = getattr(result, "names", None)
    labels, positive_ids = _positive_ids(names)
    target_ids = positive_ids or [int(cut_class_id)]

    if result is None:
        return {"is_cut_detected": False, "confidence": 0.0,
                "class_name": "no_result", "class_id": None,
                "reason": "Cut model returned no result"}

    probs = getattr(result, "probs", None)
    if probs is not None:
        values = probs.data.detach().cpu().flatten().tolist()
        if not values:
            return {"is_cut_detected": False, "confidence": 0.0,
                    "class_name": "no_probabilities", "class_id": None,
                    "reason": "Cut model returned empty probabilities"}
        predicted_id = max(range(len(values)), key=values.__getitem__)
        confidence = float(values[predicted_id])
        label = labels.get(predicted_id, str(predicted_id))
        is_cut = predicted_id in target_ids and confidence >= float(min_conf)
        return {"is_cut_detected": is_cut, "confidence": confidence,
                "class_name": label, "class_id": predicted_id,
                "reason": "positive cut class" if is_cut else "model did not select the positive cut class"}

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        class_ids = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        index = max(range(len(confidences)), key=confidences.__getitem__)
        class_id = int(class_ids[index])
        confidence = float(confidences[index])
        is_cut = class_id in target_ids and confidence >= float(min_conf)
        return {"is_cut_detected": is_cut, "confidence": confidence,
                "class_name": labels.get(class_id, str(class_id)),
                "class_id": class_id,
                "reason": "positive cut detection" if is_cut else "strongest detection is not the positive cut class"}

    return {"is_cut_detected": False, "confidence": 0.0,
            "class_name": "no_detection", "class_id": None,
            "reason": "No cut-class detection in model output"}
