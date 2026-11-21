"""Original pickup visibility and jewel-selection rules, unchanged."""
HAND = {'IN_HAND', 'UNDER_HAND'}
PICKUP_STATES = {'IN_HAND'}

def visibility(track, timestamp, bench_moving):
    """Use the same evidence as the tracker's hand labels, never absence alone."""
    if track.get('in_hand'):
        return 'IN_HAND'
    if track.get('spot') == 'hand':
        return 'UNDER_HAND'
    if track['state'] in ('MISSING', 'LEFT_ROI'):
        return 'MISSED'
    if track['state'] == 'OK' and 0 <= timestamp-track['det_seen'] <= .75:
        return 'VISIBLE'
    if track.get('spot') == 'removed' and bench_moving:
        return 'IN_HAND'
    return 'NOT_SEEN'


def choose_hand_jewel(tracks, timestamp, bench_moving, recent_hand):
    held=[tr for tr in tracks if not tr.get('acid_done')
          and visibility(tr,timestamp,bench_moving) in PICKUP_STATES]
    if held:
        return held[0]['id'] if len(held)==1 else None
    candidates = [tr for tr in tracks if not tr.get('acid_done')
                  and (visibility(tr, timestamp, bench_moving) in PICKUP_STATES
                       or 0 <= timestamp-recent_hand.get(tr['id'], -1e9) <= 2)]
    return candidates[0]['id'] if len(candidates) == 1 else None

