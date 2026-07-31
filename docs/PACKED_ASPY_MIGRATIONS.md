# Strict packed-AsPy qualification paths

This snapshot publishes two isolated Scheme A qualification identities:

- Scheme A whole-block 2% communication;
- Scheme A coordinate iso-wire 2% communication.

Both configurations explicitly require:

```yaml
model:
  execution_backend: packed_aspy
  execution_backend_strict: true
```

Strict mode requires an Ascend NPU and exactly six requested/actual native
AsPy decay-LIF routes in both training and evaluation. Any fallback is a
failure. Diagnostic activity/eligibility forwards intentionally remain eager.

## Qualified stack

- Ascend 910B4
- CANN 8.5.0
- Python 3.10.20
- PyTorch and torch-npu 2.9.0
- `spikingjelly_npu` commit `289c2ac82c759da0a01e7fd798972cb17f2f6e9b`

Do not use CANN 9.1 beta paths or `npugraph` for these qualifications.

## Commands

Use a fresh output track and an idle device. Source the environment before
setting the device variable so the script receives it:

```bash
source scripts/env_myserver.sh
export ASCEND_DEVICE_ID=<device>

python -m fedsnn.train_topk \
  --config configs/experiments/cifar10_t4_block_dual_ef_dir_v1/packed_aspy_qualification/block_2pct_seed2.yaml \
  --data-root /path/to/cifar10 \
  --device npu:<device> --smoke

python -m fedsnn.train_topk \
  --config configs/experiments/cifar10_t4_block_dual_ef_dir_v1/packed_aspy_qualification/coordinate_2pct_seed2.yaml \
  --data-root /path/to/cifar10 \
  --device npu:<device> --smoke
```

These qualifications do not change the project-wide formal backend policy and
do not rewrite historical packed-eager results.

## Evidence and identity boundary

The accepted private execution evidence used frozen FedSNN snapshot commit
`ee76b308dc4d7e4c07118863a86293599070f0d8` and the dependency commit listed
above. Each Scheme A Block and Coordinate repetition emitted one finite metric
row. Every accepted run reported six requested/actual native AsPy routes in
training, six in evaluation, and zero fallback.

This public repository is a later allowlisted publication identity. It removes
machine paths and private session metadata from the qualification configs and
applies lint-only source formatting; it does not claim that the public Git
commit itself was the commit executed for the recorded NPU smoke evidence.
