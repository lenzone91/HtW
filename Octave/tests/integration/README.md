# Integration Tests README

This folder contains Octave integration tests.

Integration tests should validate contracts across subsystem boundaries.

Current integration scope:

- load a plain YAML run config;
- build the DataModule and Lightning module through factories;
- run a tiny Lightning `fast_dev_run` fit.
- launch a short non-`fast_dev_run` debug run;
- verify execution reports, checkpoints, and scalar metric files.
