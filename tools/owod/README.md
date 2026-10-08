# Official M/S-OWODB workflow

See [the active author protocol](../../docs/owod-benchmarks.md).

1. `manage_data.py plan/check/prepare/verify`: own experiment data preparation and verification.
2. `prepare_benchmarks.py`: internal serializer for released ImageSets and original annotations.
3. `prepare_protocol.py`: register already-serialized author annotations with an image root.
4. `run_benchmark.py`: preflight and run four tasks with separate balanced finetuning.

The released lists and their pinned hashes are under `configs/owod/`.
Custom/pilot manifests and unverified overrides are rejected. Historical setting
files and launchers have been deleted. Read-only historical reporting lives in
`tools/analysis/`; model components remain in `models/`.

See [the data guide](../../docs/data-preparation.md) for raw sources and commands.
