# Integration Tests README

This folder contains Octave integration tests.

Integration tests should validate contracts across subsystem boundaries.

Current integration scope:

- load a plain YAML run config;
- build the DataModule and Lightning module through factories;
- run a tiny Lightning `fast_dev_run` fit.
