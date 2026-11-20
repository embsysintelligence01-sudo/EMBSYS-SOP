# Preserved development tools and assets

The production launcher continues to use `engines/Jeweltracker/jewel_tracker_5.py` and `engines/Packetsealing_001/packet_app.py`. These files are not selected by the production launcher.

`legacy/` holds the historical tracker versions, the complete 2026-10-27 update snapshot (including its three useful test modules and private camera defaults), and the old packet pickup snapshot. Source was moved byte-for-byte. Version 3 was an unreferenced duplicate of version 2; its separate local camera configuration was preserved beside the production engine.

List or run a supported preserved standalone script:

```bash
./development/run.sh --list
./development/run.sh pickup_snapshot.py --self-test
./development/run.sh jewel_tracker_6.py
```

The runner uses the existing SLIC interpreter, the same exclusive process lock as the main launcher, and the original engine directory as the script's model/config context. It does not rewrite the archived algorithms. The last example opens a legacy camera UI: use it only for deliberate development after the main SOP has finished. Legacy tools do not implement the current continuous surveillance contract and are never invoked automatically by EMBSYS-SOP. Historical snapshot tests run independently, using their own vintage tracker/identity modules with current shared dependencies; pre-existing failures are recorded in the cleanup report.

`models/` preserves candidate assets without active/default filename references: the hand-segmentation IR bundle and older mouth classifier/validation metadata. Models remain local and ignored by Git. No required inference or optional training/evaluation model was removed.

`fixtures/Packetsealing_001/sample_images/` contains the six original sample images. An engine-relative compatibility link preserves the optional mouth-lab file picker and existing example commands. These local images are ignored by Git.

`benchmarks/` and `editor/` retain historical performance output and editor workspace settings for reference. The editor workspace is historical, not a launcher for the combined project.
