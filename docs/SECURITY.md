# Security review

Embedded camera passwords and camera addresses were found in current and historical Jeweltracker source defaults and RTSP examples. The historical similarity app also contained a literal camera URL; it is retained in ignored `similarity_camera.local.json`. These defaults were moved to ignored, mode-0600 `*.local.json` files in the new tree; saved `camera.json` remains local and ignored. Committable defaults have blank connection fields. Safe example configurations use placeholders. Literal credential-bearing RTSP examples were sanitized; dynamic URL construction is preserved.

The initial source-inspection command printed embedded camera defaults into this session's tool output before sanitization. Treat those passwords as exposed and rotate them. Subsequent scans report locations/results without printing credentials.

Rotate the camera account passwords used in the original `jewel_tracker_5.py` and historical tracker versions, including `tracker_update_20261027/jewel_tracker_5.py`. Previously published Git history may contain them. The source repositories were deliberately left untouched, so they still contain the original defaults. This integration does not rewrite that history or change camera passwords. Update the new local `camera.json` and private default JSON files after rotation.

The root ignore file excludes environments, models, local configuration, camera JSON, evidence, logs, generated media and common secret files. Engine console output is not forwarded into launcher UI/logs because third-party camera errors may contain connection details. Diagnostic validation output is private, ignored and mode 0600. No credentials are included in this report.

Original Windows camera network provisioning helpers remain historical optional tools. Their static network examples are not credentials and are not used by the Ubuntu launcher. Do not execute provisioning helpers without configuring their placeholders for the intended host.

A static secret scan checks candidate Git files for known migrated password values and literal RTSP credentials before staging is considered safe. This is not an exhaustive credential-discovery guarantee. Recheck new changes before publishing. No GitHub download, remote configuration, commit or push is performed by installation or validation.
