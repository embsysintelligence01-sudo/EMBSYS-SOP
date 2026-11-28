# EMBSYS Packet-Cut — Automated Application

> Engine 3 now defaults to continuous pass-by observation. Missing testbed,
> OCR, strip, cutting or mouth evidence does not hold up other checks. The
> numbered stages below describe the validators, not a required release order.
> The sequential policy remains available only as `AutoPipelineConfig(pass_by=False)`.


Fully automated, event-driven desktop application for secure packet cutting, post-cutting inspection, and appraiser workflow compliance.

## Key Capabilities

1. **Operator Controls**: Load the models and check the camera, draw the testbed ROI, then start the automated inspection. All inspection models receive only the selected region. The live preview still shows the full camera frame. Retry the current stage as needed. Stop the session to save and copy the compliance report.
2. **Event-Driven Stage Transitions**:
   - Stage 1: Empty testbed detection (debounced across fresh frames)
   - Stage 2: Packet frontside detection and PaddleOCR packet number extraction.
   - Stage 3: Packet flip and Top Strip verification. A verified absent strip shows **READY TO CUT**.
   - Stage 4: Detect the hand and scissors together, then move to Stage 5 as soon as both leave the selected region.
   - Stage 5: Analyze the cut for 10 frames using the cut model, two detached pieces, and the gap between them.
   - Stage 6: Present the packet mouth upright for 10 frames of analysis. Confirm **OPEN** or **CLOSED** and verify gold visibility for an open mouth.
   - Stage 7A: Wait for confirmed gold placement on the testbed.
   - Stage 7B: Track and count jewellery on the testbed until **STOP** saves the report.
3. **Appraiser Procedure Monitoring**: Continuous SOP monitoring detecting skipped or out-of-order steps without blocking physical inspection.
4. **Audit Trail**: Generates timestamped activity logs and compliance reports in `packet_cut_outputs/automated_sessions/`.

## Running the Application

### 1. Main Automated Inspection App (PyQt6)
```powershell
# From project root:
.\.venv\Scripts\python.exe automated_app/app.py

# With specific camera:
.\.venv\Scripts\python.exe automated_app/app.py --camera 0
```

### 2. Standalone Compliance & SOP Monitoring Window
```powershell
# From project root:
.\.venv\Scripts\python.exe automated_app/app_compliance.py
```

### 3. Automated Test Suites
```powershell
# Run all automated tests:
.\.venv\Scripts\python.exe -m unittest discover -s automated_app/tests -p "test_*.py"
```

## Directory Structure
- `packet_cut/`: Desktop application, state machine, worker threads, and automation pipeline.
- `inspection/`: Post-cutting mouth opening inspection and jewellery counting modules.
- `workflow_monitor/`: Compliance tracker, alert engine, activity logger, and report generator.
- `tests/`: Automated unit and integration test suites.
- `packet_cut_core.py`: Computer vision engine (piece extraction, unpadded masks, gap analysis).



## Ubuntu mini PC

Run from the project root:

```bash
./run_automated.sh --camera 0
```

The launcher uses `.venv-ubuntu` (Python 3.12), CPU inference, four CPU threads,
and the English PaddleOCR mobile detection and recognition models. It resolves
model paths from the project folder, so it also works when called from another
directory. Select **Load Models / Check Ready**, then **Draw ROI**, then **START**.
Use **STOP** to save the report before closing.

The copied `.venv` contains Windows binaries and is not used on Linux.
`requirements-linux.lock` records the installed Linux package versions. To rebuild
with Python 3.12 available:

```bash
python3.12 -m venv .venv-ubuntu
.venv-ubuntu/bin/python -m pip install --upgrade pip
.venv-ubuntu/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux.lock
```

The seven vision models are in `models/`. OCR weights are cached separately in
`~/.paddlex/official_models/`; a new computer needs Internet access on its first
OCR startup. A connected V4L2 camera is required for live inspection.


### Fixed testbed markings

Before checking readiness, remove packets, hands, jewellery and loose objects.
Leave permanent markings, rectangles of any colour, labels and ArUco markers in
place. The preview selected with **Draw ROI** becomes the empty background
reference. After **START**, unchanged background markings are ignored in both
the empty-bed check and subsequent packet detection. Newly placed objects are
still checked. Small global lighting changes are compensated automatically.

Keep the camera and mat fixed after setup. If either moves or the markings change,
stop, check readiness again, and redraw the ROI to capture a fresh empty reference.
Objects present in the selected reference are treated as background, so capture
it only with the surface empty of loose objects.


## Visual reports and short evidence clips

Press **STOP** to save the session, then **Open Visual Report**. Each session folder
now contains an HTML report alongside JSON and Markdown. The HTML report shows the
packet number image and number crop, verified cut image, packet mouth image, and
jewellery image with article labels and the reported count/status. Missing evidence
is labelled explicitly; provisional and uncertain counts are not promoted to final
verified counts. OCR identity resolution does not establish a database ID match.

`clips/` contains event clips for packet-number recognition, confirmed cut, mouth
inspection and the latest jewellery count. Capture reuses inspected ROI frames:
four seconds before and three seconds after the event, up to 640x480, 5 fps,
WebM/VP8, no audio. Clips include packet ID, event result/count and timestamps.
A bounded JPEG ring buffers recent frames; the full session is not recorded.
STOP flushes a shorter clip immediately. When inference is slow, the last captured
frame is held between samples to preserve elapsed time. Capture or codec failures
are included in the report and do not claim that video was saved.

Images are in `evidence/`; clip metadata is in the JSON report's `evidence_media`
list. Keep the session folder together when copying/sharing it so relative links
continue to work. HTML can be printed from the browser; videos remain linked files.
Existing sessions can use their saved images, but missing historical videos cannot
be recreated. Restart the application after installing this change.


### Engine 3 PDF export

**STOP** also creates an image-based PDF in the session folder. Use **Open PDF
Report** in the app to view it. The PDF includes session/packet identity, timestamps,
stage reached, packet-number and cut images, mouth inspection, jewellery count with
its provisional/uncertain status, and warnings. Images are embedded; video clips
remain separate files playable through the HTML report. The PDF is an inspection
evidence report, not an automated physical release authorization. PDF export errors
are recorded without discarding JSON, Markdown, or HTML evidence. Install the
`reportlab` dependency when setting up a new environment.

The default release policy evaluates checks independently and records missing or
late observations. The optional strict policy waits for each stage. Neither policy
integrates a separate purity, sealing, database matching, or physical release controller.


### Continuous release observations (default)

Select the work area and START at any point, including with a packet already on
it. Empty-bed verification is an observation, not an entry requirement. The Draw
ROI dialog offers optional calibration only when its preview is known empty;
leave that option unchecked for an occupied preview. Unknown models are reported
as unavailable checks rather than preventing other available checks from running.
A usable camera and selected work area are still needed to capture evidence.

OCR, seal/backside, cutting, physical cut, mouth, placement and counting checks run
independently. Available evidence is validated using the existing physical rules;
missing evidence is never converted to a pass. OCR can resolve later without
resetting counting, and the audit records that observation as late. Seeing no strip
alone does not prove that the packet is sealed or that its backside was shown.

The live display separates the visible count from a retained clear count. Retained
count means the highest simultaneously visible unambiguous count stable across
three observation frames; it is not a sum of articles seen over time and is not
proof all contents were removed. Removing articles, hands obscuring the view, or
brief detections do not erase that evidence. A session without a stable positive
count saves null/NOT_VERIFIED rather than a verified zero. Start a new session for
a different packet. STOP always saves available evidence and a release-check table
in PDF, HTML and JSON; unverified checks and late recoveries are listed explicitly.


### Live preview and CPU inference budget

Live USB/network camera capture runs independently of inspection and keeps only
the newest frame. Preview targets 15 fps; the system panel displays the measured
preview and analysis rates separately. Live preview shows the current camera view
and selected work area; numbered inspection overlays remain in saved evidence
images. Recorded video files/frame providers keep sequential analysis behavior.

After the ID is locked from two fresh number crops, both number-region detection
and PaddleOCR inference stop for that session. Before locking, number checks run
at most every 0.75 seconds and OCR runs only on a validated sharp crop, using one
preprocessing pass per fresh attempt. Crop upscaling is bounded to 960x256.
Unresolved identification remains eligible for later observations.

Jewellery inference is skipped while hands obscure the area, scissors stop after
cutting and cut separation are verified, and remaining checks are sampled rather
than all executed on every analysis frame. Skipped frames never count toward
verification or tracking confirmation. Previous clear count evidence is retained.
Event clips buffer bounded JPEG windows during inspection and encode at STOP to
avoid live video-encoding stalls. These changes retain the continuous pass-by
policy; missing observations still do not block the other checks.


### Cutting activity display and evidence

Cutting activity (hand and scissors together) and confirmed cut separation are
separate report sections. The activity snapshot and short clip are retained even
when separation never reaches CUT_CONFIRMED. Available cut candidates are also
saved as CUT_UNVERIFIED, with the analysed/stable frame counts and the reason full
confirmation is pending. This does not relax the physical verification thresholds.

The smooth main live camera remains current. A small timestamped last-analysis
inset shows hand/scissor masks and labels on the exact analysed frame, rather than
projecting stale masks onto newer camera frames. It expires after 1.5 seconds and
clears when detections clear. The live cutting field distinguishes detected/current
activity from a cutting event recorded earlier in the session. No extra model
inference is needed for this display.

Older reports can show an existing cutting photo from their evidence folder even
if their original media list omitted it. Missing historical video or cut-candidate
images are not fabricated; new captures require a new inspection session.

### Fast packet-number OCR

Pass-by release uses PaddleOCR's standalone mobile text recognizer on the
number detector's crop, avoiding a second text-location pass. Number checks are
scheduled at 0.35-second intervals and require two stable, sharp views before
reading; two matching fresh readings at >=65% confidence still lock the ID.
The original three-view setting remains for strict sequential mode. A missing
or moving number region still resets stability. After ID lock, number detection
and OCR stop for that session.

If direct recognition fails, every third failed attempt may use one bounded
full OCR pass to recover difficult layouts. This fallback stays inside the
number crop. Model loading prepares the mobile recognizer; the full OCR
pipeline is loaded only if needed. Session JSON records number-detector and
OCR call counts and elapsed milliseconds under
`operational_results.ocr_identification.timing` (OCR includes preprocessing and
any fallback/model initialization within the call).

A local replay of the saved 324871 number crop read the correct ID at 99.6%
confidence: warm direct recognition averaged 50.6 ms versus 72.9 ms with text
detection plus recognition (three runs each). These are crop timings, not a
live-camera lock-time guarantee; initialization and all other inference add
latency.
