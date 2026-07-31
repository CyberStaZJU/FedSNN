from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SHD_CACHE_SCHEMA = "fedsnn.shd.binned.v1"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "name": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = _sha256(path)
    return identity


class SHDH5Dataset:
    """Lazily read SHD HDF5 and bin events using one handle per process.

    No HDF5 handle is opened until ``__getitem__``. A DataLoader worker that is
    forked or spawned therefore opens and reuses its own handle rather than
    inheriting a live handle from the parent process.
    """

    input_units = 700
    classes = 20

    def __init__(
        self,
        path: str | Path,
        *,
        timesteps: int,
        duration: float = 1.0,
        binary: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"SHD file does not exist: {self.path}")
        if timesteps <= 0 or not np.isfinite(duration) or duration <= 0:
            raise ValueError("timesteps and duration must be positive")
        self.timesteps = int(timesteps)
        self.duration = float(duration)
        self.binary = bool(binary)
        self._handle = None
        self._handle_pid = None
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - runtime environment only
            raise RuntimeError("h5py is required to read SHD") from exc
        with h5py.File(self.path, "r") as handle:
            required = ("spikes/times", "spikes/units", "labels")
            missing = [key for key in required if key not in handle]
            if missing:
                raise ValueError(f"invalid SHD file; missing datasets: {missing}")
            self.targets = np.asarray(handle["labels"], dtype=np.int64)
            if len(handle["spikes/times"]) != len(self.targets):
                raise ValueError("SHD spikes and labels have different lengths")
            self.available_metadata = tuple(sorted(handle.keys()))
            speaker_key = next(
                (
                    key
                    for key in ("extra/speaker", "speakers", "speaker", "speaker_ids")
                    if key in handle
                ),
                None,
            )
            self.speaker_ids = (
                np.asarray(handle[speaker_key], dtype=np.int64)
                if speaker_key is not None
                else None
            )
            if self.speaker_ids is not None and len(self.speaker_ids) != len(self.targets):
                raise ValueError("SHD speakers and labels have different lengths")
            self.speaker_key = speaker_key
            self.has_speaker_ids = self.speaker_ids is not None

    def __len__(self) -> int:
        return int(len(self.targets))

    def _h5_handle(self):
        import h5py

        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._handle_pid = pid
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._handle_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __getitem__(self, index: int):
        import torch

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        handle = self._h5_handle()
        times = np.asarray(handle["spikes/times"][index], dtype=np.float64)
        units = np.asarray(handle["spikes/units"][index], dtype=np.int64)
        if times.shape != units.shape:
            raise ValueError(f"SHD event arrays disagree at sample {index}")
        valid = (
            np.isfinite(times)
            & (times >= 0)
            & (times < self.duration)
            & (units >= 0)
            & (units < self.input_units)
        )
        bins = np.floor(times[valid] * self.timesteps / self.duration).astype(np.int64)
        flat = bins * self.input_units + units[valid]
        values = np.zeros(self.timesteps * self.input_units, dtype=np.float32)
        if self.binary:
            values[np.unique(flat)] = 1.0
        else:
            np.add.at(values, flat, 1.0)
        events = torch.from_numpy(values.reshape(self.timesteps, self.input_units))
        return events, int(self.targets[index])


class SHDBinnedDataset:
    """Read a validated, binary SHD cache through a process-local NumPy mmap."""

    input_units = 700
    classes = 20

    def __init__(self, cache_dir: str | Path, split: str, manifest: dict[str, Any]) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"unsupported SHD split: {split}")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.timesteps = int(manifest["binning"]["timesteps"])
        self.duration = float(manifest["binning"]["duration_seconds"])
        self.binary = bool(manifest["binning"]["binary"])
        if not self.binary:
            raise ValueError("SHD binned cache v1 supports binary bins only")
        entry = manifest["splits"][split]
        self.events_path = self.cache_dir / entry["events_file"]
        self.targets = np.load(self.cache_dir / entry["labels_file"], allow_pickle=False)
        speaker_file = entry.get("speakers_file")
        self.speaker_ids = (
            np.load(self.cache_dir / speaker_file, allow_pickle=False)
            if speaker_file is not None
            else None
        )
        self.speaker_key = manifest.get("speaker_dataset_key")
        self.has_speaker_ids = self.speaker_ids is not None
        self.available_metadata = tuple(manifest.get("hdf5_root_keys", ()))
        self._events = None
        self._events_pid = None
        expected_shape = (len(self.targets), self.timesteps, self.input_units)
        if tuple(entry["shape"]) != expected_shape or entry["dtype"] != "uint8":
            raise ValueError(f"invalid cached SHD {split} shape or dtype")
        if self.speaker_ids is not None and len(self.speaker_ids) != len(self.targets):
            raise ValueError("cached SHD speakers and labels have different lengths")

    def __len__(self) -> int:
        return int(len(self.targets))

    def _event_array(self):
        pid = os.getpid()
        if self._events is None or self._events_pid != pid:
            self._events = np.load(self.events_path, mmap_mode="r", allow_pickle=False)
            self._events_pid = pid
        return self._events

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_events"] = None
        state["_events_pid"] = None
        return state

    def __getitem__(self, index: int):
        import torch

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # The float32 copy preserves the original trainer input dtype and avoids
        # exposing a read-only mmap buffer to torch.
        events = np.asarray(self._event_array()[index], dtype=np.float32)
        return torch.from_numpy(events), int(self.targets[index])


def resolve_shd_files(root: str | Path) -> tuple[Path, Path]:
    """Resolve extracted SHD train/test files from an external dataset root."""

    root = Path(root)
    candidates = (root, root / "raw", root / "shd", root / "shd" / "raw")
    for directory in candidates:
        train = directory / "shd_train.h5"
        test = directory / "shd_test.h5"
        if train.is_file() and test.is_file():
            return train, test
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"SHD train/test HDF5 files not found under: {searched}")


def _validate_cache(
    cache_dir: Path,
    train_path: Path,
    test_path: Path,
    *,
    timesteps: int,
    duration: float,
    binary: bool,
    expected_manifest_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SHD cache manifest does not exist: {manifest_path}")
    manifest_sha256 = _sha256(manifest_path)
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("SHD cache manifest SHA256 does not match the frozen config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SHD_CACHE_SCHEMA:
        raise ValueError("unsupported SHD cache schema")
    binning = manifest.get("binning", {})
    if (
        int(binning.get("timesteps", -1)) != int(timesteps)
        or not np.isclose(float(binning.get("duration_seconds", -1)), float(duration))
        or bool(binning.get("binary")) != bool(binary)
    ):
        raise ValueError("SHD cache binning does not match requested semantics")
    for split, source_path in (("train", train_path), ("test", test_path)):
        source = manifest["sources"][split]
        current = _source_identity(source_path, include_sha256=False)
        for key in ("name", "size_bytes", "mtime_ns"):
            if source.get(key) != current[key]:
                raise ValueError(f"SHD cache source identity mismatch for {split}: {key}")
        entry = manifest["splits"][split]
        for key in ("events_file", "labels_file"):
            if not (cache_dir / entry[key]).is_file():
                raise FileNotFoundError(f"SHD cache is missing {entry[key]}")
        if entry.get("speakers_file") and not (cache_dir / entry["speakers_file"]).is_file():
            raise FileNotFoundError(f"SHD cache is missing {entry['speakers_file']}")
    return manifest, manifest_sha256


def load_shd(
    root: str | Path,
    *,
    timesteps: int,
    duration: float = 1.0,
    binary: bool = True,
    cache_dir: str | Path | None = None,
    require_cache: bool = False,
    expected_cache_manifest_sha256: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    train_path, test_path = resolve_shd_files(root)
    manifest = None
    manifest_sha256 = None
    resolved_cache = Path(cache_dir) if cache_dir is not None else None
    if resolved_cache is not None:
        if not resolved_cache.is_absolute():
            resolved_cache = Path(root) / resolved_cache
        try:
            manifest, manifest_sha256 = _validate_cache(
                resolved_cache,
                train_path,
                test_path,
                timesteps=timesteps,
                duration=duration,
                binary=binary,
                expected_manifest_sha256=expected_cache_manifest_sha256,
            )
        except (FileNotFoundError, ValueError):
            if require_cache:
                raise
            manifest = None
    elif require_cache:
        raise ValueError("require_cache=true but no SHD cache directory was configured")

    if manifest is not None:
        train = SHDBinnedDataset(resolved_cache, "train", manifest)
        test = SHDBinnedDataset(resolved_cache, "test", manifest)
        source = "Zenke Lab SHD pre-binned mmap cache"
        data_backend = "npy_mmap_uint8"
    else:
        train = SHDH5Dataset(train_path, timesteps=timesteps, duration=duration, binary=binary)
        test = SHDH5Dataset(test_path, timesteps=timesteps, duration=duration, binary=binary)
        source = "Zenke Lab SHD HDF5"
        data_backend = "process_local_hdf5"
    metadata = {
        "source": source,
        "data_backend": data_backend,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "cache_dir": str(resolved_cache) if manifest is not None else None,
        "cache_manifest_sha256": manifest_sha256,
        "timesteps": int(timesteps),
        "duration_seconds": float(duration),
        "binary_bins": bool(binary),
        "hdf5_root_keys": list(train.available_metadata),
        "speaker_ids_available": bool(train.has_speaker_ids),
        "speaker_dataset_key": train.speaker_key,
        "train_speaker_count": (
            int(np.unique(train.speaker_ids).size) if train.speaker_ids is not None else 0
        ),
        "test_speaker_count": (
            int(np.unique(test.speaker_ids).size) if test.speaker_ids is not None else 0
        ),
    }
    return train, test, metadata


def build_shd_cache(
    root: str | Path,
    cache_dir: str | Path,
    *,
    timesteps: int,
    duration: float = 1.0,
    binary: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> Path:
    """Build an atomic uint8 cache through process-safe HDF5 DataLoader workers."""

    import shutil
    import time
    from torch.utils.data import DataLoader

    if not binary:
        raise ValueError("SHD cache v1 supports binary bins only")
    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise ValueError("invalid SHD cache worker settings")
    train_path, test_path = resolve_shd_files(root)
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing SHD cache: {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = cache_dir.parent / f".{cache_dir.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"SHD cache staging path already exists: {staging}")
    staging.mkdir()
    started = time.time()
    manifest: dict[str, Any] = {
        "schema": SHD_CACHE_SCHEMA,
        "binning": {
            "timesteps": int(timesteps),
            "duration_seconds": float(duration),
            "binary": True,
            "input_units": 700,
        },
        "generator": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "persistent_workers": bool(num_workers > 0),
            "prefetch_factor": int(prefetch_factor) if num_workers > 0 else None,
        },
        "sources": {},
        "splits": {},
    }
    try:
        for split, source_path in (("train", train_path), ("test", test_path)):
            dataset = SHDH5Dataset(
                source_path, timesteps=timesteps, duration=duration, binary=True
            )
            if split == "train":
                manifest["hdf5_root_keys"] = list(dataset.available_metadata)
                manifest["speaker_dataset_key"] = dataset.speaker_key
            events_name = f"{split}_events.npy"
            labels_name = f"{split}_labels.npy"
            speakers_name = f"{split}_speakers.npy" if dataset.speaker_ids is not None else None
            events = np.lib.format.open_memmap(
                staging / events_name,
                mode="w+",
                dtype=np.uint8,
                shape=(len(dataset), timesteps, dataset.input_units),
            )
            loader_kwargs: dict[str, Any] = {
                "batch_size": int(batch_size),
                "shuffle": False,
                "drop_last": False,
                "num_workers": int(num_workers),
                "persistent_workers": bool(num_workers > 0),
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            loader = DataLoader(dataset, **loader_kwargs)
            offset = 0
            for batch, _labels in loader:
                count = int(batch.shape[0])
                values = batch.numpy()
                if not np.all((values == 0) | (values == 1)):
                    raise ValueError("binary SHD cache encountered a non-binary bin")
                events[offset : offset + count] = values.astype(np.uint8, copy=False)
                offset += count
            if offset != len(dataset):
                raise RuntimeError(f"SHD cache wrote {offset}/{len(dataset)} {split} samples")
            events.flush()
            del events
            np.save(staging / labels_name, np.asarray(dataset.targets, dtype=np.int64))
            if speakers_name is not None:
                np.save(staging / speakers_name, np.asarray(dataset.speaker_ids, dtype=np.int64))
            manifest["sources"][split] = _source_identity(source_path, include_sha256=True)
            manifest["splits"][split] = {
                "events_file": events_name,
                "labels_file": labels_name,
                "speakers_file": speakers_name,
                "shape": [len(dataset), timesteps, dataset.input_units],
                "dtype": "uint8",
            }
        for entry in manifest["splits"].values():
            files = [entry["events_file"], entry["labels_file"]]
            if entry.get("speakers_file"):
                files.append(entry["speakers_file"])
            entry["files"] = {
                name: {
                    "size_bytes": (staging / name).stat().st_size,
                    "sha256": _sha256(staging / name),
                }
                for name in files
            }
        manifest["elapsed_seconds"] = time.time() - started
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, cache_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return cache_dir / "manifest.json"


class ClassSubsetSHD:
    """View over an SHD split restricted to a contiguous class subset.

    Labels are remapped to ``0 .. K-1`` in sorted kept-class order so a K-way
    classifier can train without changing the underlying binned cache.  Used by
    literature-aligned simple-task probes (e.g. first five SHD digits), not by
    the frozen full-class Stage-1A protocol.
    """

    def __init__(self, base: Any, keep_classes: Sequence[int]) -> None:
        if not keep_classes:
            raise ValueError("keep_classes must be non-empty")
        self.base = base
        self.keep_classes = tuple(int(c) for c in keep_classes)
        if len(set(self.keep_classes)) != len(self.keep_classes):
            raise ValueError("keep_classes must be unique")
        if min(self.keep_classes) < 0:
            raise ValueError("keep_classes must be nonnegative")
        mapping = {src: dst for dst, src in enumerate(self.keep_classes)}
        base_targets = np.asarray(base.targets, dtype=np.int64)
        mask = np.isin(base_targets, np.asarray(self.keep_classes, dtype=np.int64))
        self.indices = np.flatnonzero(mask).astype(np.int64)
        if self.indices.size == 0:
            raise ValueError(f"no samples for keep_classes={self.keep_classes}")
        remapped = np.empty(self.indices.shape[0], dtype=np.int64)
        for position, source_index in enumerate(self.indices):
            remapped[position] = mapping[int(base_targets[source_index])]
        self.targets = remapped
        self.input_units = int(getattr(base, "input_units", 700))
        self.classes = int(len(self.keep_classes))
        # Preserve optional metadata used by diagnostics when present.
        for attr in ("timesteps", "duration", "binary", "speaker_ids", "has_speaker_ids"):
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        events, _raw_label = self.base[int(self.indices[index])]
        return events, int(self.targets[index])
