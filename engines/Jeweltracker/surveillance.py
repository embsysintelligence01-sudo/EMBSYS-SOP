"""Evidence labels and immutable handoff data, independent of physical progress."""
import copy
import time


def identity_status(track, now=None):
    if getattr(track, 'id', None) is None:
        return 'UNASSIGNED'
    if (getattr(track, 'identity_uncertain', False)
            or getattr(track, 'identity_status', '') in ('UNCERTAIN', 'UNASSIGNED')
            or getattr(track, 'state', 'OK') != 'OK'
            or (time.time() if now is None else now) - getattr(track, 'det_seen', 0) > 2.5):
        return 'UNCERTAIN'
    return 'ASSIGNED'


def purity_evidence(track):
    record = copy.deepcopy(getattr(track, 'acid_result', None) or {})
    prediction = record.get('prediction')
    valid = bool(prediction and prediction.get('prediction') in ('gold_22k', 'non_gold'))
    rubbing = bool(getattr(track, 'rubbed', False))
    acid = bool(getattr(track, 'acid_done', False))
    mask = record.get('mask_evidence_status', 'UNCONFIRMED')
    recovery = copy.deepcopy(record.get('backup') or {})
    verified = bool(valid and rubbing and acid and mask == 'VERIFIED'
        and record.get('sop_review_status') == 'VERIFIED'
        and record.get('acid_target_status') == 'ACID_OVERLAPS_NEW_RUB'
        and record.get('reaction_coverage_status') == 'COMPLETE'
        and not recovery.get('timed_out') and (not recovery or recovery.get('status') == 'VERIFIED'))
    status = 'VERIFIED' if verified else 'REVIEW_REQUIRED' if prediction else 'INCOMPLETE'
    return dict(id=getattr(track, 'id', None), identity_status=identity_status(track),
        rubbed=rubbing, acid_done=acid, prediction=prediction,
        close_reason=record.get('close_reason'), test_status=status, evidence_status=status,
        rubbing_status=record.get('rubbing_status', 'CONFIRMED' if rubbing else 'UNCONFIRMED'),
        mask_status=mask, acid_application_status=record.get('fresh_acid_status', 'UNCONFIRMED'),
        classification_status='CONFIRMED' if valid else 'INVALID' if prediction else 'MISSING',
        recovery=recovery, record_path=record.get('record_path'),
        capture_finished=bool(getattr(track, 'test_captured', False) or record.get('close_reason') or prediction))


def aggregate_purity(jewels):
    statuses = [j.get('evidence_status', 'INCOMPLETE') for j in jewels]
    if not statuses or 'INCOMPLETE' in statuses:
        return 'INCOMPLETE'
    return 'VERIFIED' if all(s == 'VERIFIED' for s in statuses) else 'REVIEW_REQUIRED'


def retain_auxiliary_result(value, directory):
    """Keep completed manual rub evidence without declaring it verified."""
    import json
    import os
    from pathlib import Path
    import numpy as np
    directory = Path(directory) / 'handoff_auxiliary'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    arrays = []
    def encode(item):
        if isinstance(item, np.ndarray):
            name = f'array_{len(arrays)}.npy'
            arrays.append(name)
            with (directory / name).open('wb') as handle:
                os.chmod(directory / name, 0o600)
                np.save(handle, item, allow_pickle=False)
            return {'array': name}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): encode(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(val) for val in item]
        if isinstance(item, Path):
            return str(item)
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        # RubTarget is plain evidence data; never serialize a runtime/model.
        return {key: encode(getattr(item, key)) for key in
            ('metadata', 'evidence', 'shape', 'new_mask', 'roi_mask', 'stone', 'streak') if hasattr(item, key)}
    payload = {'evidence_status': 'REVIEW_REQUIRED', 'result': encode(value)}
    temporary = directory / 'result.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.chmod(0o600)
    temporary.replace(directory / 'result.json')
    return str(directory / 'result.json')
