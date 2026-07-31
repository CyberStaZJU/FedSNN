# Upstream provenance

This focused snapshot is a controlled FedSNN reproduction and strict
packed-AsPy integration. It is not an unrelated clean-room SNN implementation.

## FedSNN

- Repository: https://github.com/Intelligent-Computing-Lab-Panda/FedSNN
- Pinned commit: `1ab26154b064119850bc9f84c55304b5b45f7df6`
- License: MIT
- Original copyright: 2021 Yeshwanth Venkatesha

The AlexNet-BNTT implementation preserves the relevant FedSNN temporal
encoding, BNTT, membrane recurrence, detached reset, and cumulative-readout
semantics while providing explicit device/backend routing. The top-level MIT
license retains the original attribution and identifies later work as
contributor modifications.

## SpikingJelly-NPU

- Repository: https://github.com/CyberStaZJU/SpikingJelly_npu
- Required qualified commit: `289c2ac82c759da0a01e7fd798972cb17f2f6e9b`
- License: Apache-2.0

The older `v0.1.0-alpha.1` release predates the FedSNN decay-LIF native
symbols. Build the pinned commit on the qualified CANN stack.
