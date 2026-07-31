# FedSNN strict packed-AsPy qualification snapshot

This focused source snapshot contains the Scheme A trainer paths used to qualify
strict native packed-AsPy execution for FedSNN AlexNet-BNTT.

It preserves the relevant FedSNN temporal encoding, BNTT, membrane recurrence,
detached reset, and cumulative-readout semantics. See `docs/UPSTREAMS.md` for
attribution and pinned upstreams.

## Scope

- Scheme A whole-block and coordinate iso-wire Top-K paths;
- explicit `legacy_stepwise`, `packed_eager`, and strict `packed_aspy` backend
  construction for the shipped AlexNet-BNTT model;
- focused tests for configuration isolation, communication semantics, and
  backend routing.

This is not a project-wide switch to packed AsPy. Historical results retain
their original identities, and `npugraph` is not qualified for these paths.
The accepted NPU execution identity and the later public-snapshot sanitization
boundary are documented in `docs/PACKED_ASPY_MIGRATIONS.md`.

## Install and CPU checks

```bash
python -m pip install -e '.[train,test]'
python -m pytest -q
```

Native AsPy execution additionally requires an Ascend NPU, CANN 8.5.0,
PyTorch and torch-npu 2.9.0, plus `spikingjelly_npu` commit
`289c2ac82c759da0a01e7fd798972cb17f2f6e9b`. The older alpha release does
not contain the required FedSNN decay-LIF symbols.

See `docs/PACKED_ASPY_MIGRATIONS.md` for exact smoke commands and acceptance
boundaries. Runtime datasets, outputs, logs, checkpoints, queues, and caches
must be stored outside this source repository.
