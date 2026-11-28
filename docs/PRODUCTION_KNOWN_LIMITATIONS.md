# Known limitations and pending physical tests

Software validation is not bank operational acceptance or a physical loan/release
authorization. Workflow completion never makes missing evidence verified.

- Physical button/LCD/LED serial hardware is unavailable and uncommissioned.
  Serial mode fails explicitly; its transport-independent command boundary is ready.
- USB unplug/replug, actual monitor hot unplug, mains power loss/reboot and a full
  operator SOP with jewellery, cutting, sealing, rubbing and acid remain bench
  acceptance tasks. Recorded/simulated tests are labelled as such.
- Existing release mouth false positives and multi-frame cut confirmation rules
  were not changed in this architecture migration. Its proposed 320-input MobileNet
  classifier has not been supplied. Reports retain actual independent check outcomes.
- A restarted process cannot recreate live identity certainty from a serialized
  audit. Historical IDs/features remain recorded; resumed linked sealing uses the
  existing uncertain/unassigned behaviour rather than forcing a match.
- STOP preserves pending original native analysis while its host remains alive.
  Unexpected process/power loss retains raw evidence and durable publication jobs.
  A lost in-memory inference job is interrupted and requires original recovery/review;
  no unrecorded rubbing/acid/classification result is invented. Durable PDF publication
  jobs resume; an interrupted acquisition never silently restarts itself.
- Existing per-engine camera resolution, anti-flicker and ROI remain unchanged.
  The high-resolution sealing replay can report inference-too-slow during cold
  initialization; such observations remain unverified. Bench-measured throughput
  and acceptable camera configuration must be reviewed on this CPU before rollout.
- Root/system library packages are OS dependencies. The local environment copies
  remove runtime reliance on the original engine repositories, but this is not a
  portable single executable for a different Ubuntu/Python/CPU installation.
- The pre-existing OpenCV distribution metadata mismatch is recorded in the
  dependency report; engine versions were preserved instead of replacing contrib.
- Only the supported managed launches participate in the advisory camera lock.
  Arbitrary old backup/third-party scripts can bypass it; do not run them alongside
  the managed application. Already-open video devices are detected at managed start.
- Browser access uses a shared bearer token when configured, not individual bank
  employee accounts. Authenticated LAN access requires TLS and private token setup.
- Existing original archive/retention policies remain active. Common PDF copies do
  not authorize deleting source evidence. Administrative raw-evidence review remains
  outside the queue-only browser interface.
