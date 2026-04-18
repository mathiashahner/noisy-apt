#!/usr/bin/env python3
"""
Build a piano transcription fine-tuning dataset by mixing MAESTRO clean piano audio
with MUSDB18 stems (vocals/drums/bass) while preserving MAESTRO MIDI labels.

Outputs:
  - audio mixtures (.wav)
  - copied MAESTRO MIDI labels (.midi/.mid)
  - metadata.csv with mode, SNR target, used stems, source files, etc.

Mixture modes and target ratio (uniform by construction):
  25% clean
  25% noise-only (drums and/or bass)
  25% vocal-only (vocals)
  25% vocal+noise (vocals + drums and/or bass)

SNR curriculum by sample progress:
  first 30% of generated samples: SNR in [20, 5] dB
  remaining 70%:               SNR in [20, -5] dB
"""

import argparse
import csv
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


AUDIO_EXTS = {".wav"}
MIDI_EXTS = {".mid", ".midi"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_audio_file(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS


def is_midi_file(p: Path) -> bool:
    return p.suffix.lower() in MIDI_EXTS


def find_maestro_pairs(maestro_root: Path) -> List[Tuple[Path, Path]]:
    csv_candidates = sorted(maestro_root.glob("maestro-v*.csv"))
    pairs: List[Tuple[Path, Path]] = []

    maestro_csv = csv_candidates[-1]
    df = pd.read_csv(maestro_csv)

    for _, row in df.iterrows():
        a = maestro_root / str(row["audio_filename"])
        m = maestro_root / str(row["midi_filename"])
        pairs.append((a, m))

    return pairs


def index_musdb_stems(musdb_root: Path) -> Dict[str, List[Path]]:
    stems = {"vocals": [], "drums": [], "bass": []}

    for p in musdb_root.rglob("*"):
        if not (p.is_file() and is_audio_file(p)):
            continue

        name = p.name.lower()
        if "vocals" in name:
            stems["vocals"].append(p)
        elif "drums" in name:
            stems["drums"].append(p)
        elif "bass" in name:
            stems["bass"].append(p)

    return stems


def load_audio(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def tile_or_crop(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x

    if len(x) > target_len:
        start = random.randint(0, len(x) - target_len)
        return x[start : start + target_len]

    reps = math.ceil(target_len / len(x))
    y = np.tile(x, reps)[:target_len]
    return y


def sample_snr_db(progress: float) -> float:
    if progress <= 0.30:
        lo, hi = 5.0, 20.0
    else:
        lo, hi = -5.0, 20.0

    return random.uniform(lo, hi)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def scale_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale noise so that SNR(clean/noise)=snr_db.
    """
    rc = rms(clean)
    rn = rms(noise)
    desired_rn = rc / (10 ** (snr_db / 20.0))
    scale = desired_rn / (rn + 1e-12)
    return noise * scale


def choose_mode(i: int, n: int) -> str:
    """
    Enforce exact-ish 25% mode split by cycling modes.
    """
    modes = ["clean", "noise_only", "vocal_only", "vocal_noise"]
    return modes[i % 4]


def mix_interferers(
    mode: str,
    musdb_index: Dict[str, List[Path]],
    target_len: int,
    sr: int,
) -> Tuple[np.ndarray, List[Path]]:
    """
    Build interference track according to mode.
    - noise_only: drums and/or bass (1 or 2 stems)
    - vocal_only: vocals (1 stem)
    - vocal_noise: vocals + (drums and/or bass), one or more non-vocal stems
    """
    chosen_paths: List[Path] = []
    parts: List[np.ndarray] = []

    def pick_and_prepare(pool_name: str) -> Optional[np.ndarray]:
        pool = musdb_index.get(pool_name, [])
        if not pool:
            return None
        p = random.choice(pool)
        y = load_audio(p, sr)
        y = tile_or_crop(y, target_len)
        chosen_paths.append(p)
        return y

    if mode == "clean":
        return np.zeros(target_len, dtype=np.float32), chosen_paths

    if mode == "noise_only":
        candidates = [k for k in ["drums", "bass"] if len(musdb_index.get(k, [])) > 0]
        k = random.randint(1, max(1, len(candidates)))
        picked_types = random.sample(candidates, k=k)
        for t in picked_types:
            y = pick_and_prepare(t)
            if y is not None:
                parts.append(y)

    elif mode == "vocal_only":
        y = pick_and_prepare("vocals")
        if y is not None:
            parts.append(y)

    elif mode == "vocal_noise":
        yv = pick_and_prepare("vocals")
        if yv is not None:
            parts.append(yv)
        candidates = [k for k in ["drums", "bass"] if len(musdb_index.get(k, [])) > 0]
        if candidates:
            k = random.randint(1, len(candidates))
            picked_types = random.sample(candidates, k=k)
            for t in picked_types:
                yn = pick_and_prepare(t)
                if yn is not None:
                    parts.append(yn)

    inter = np.sum(np.stack(parts, axis=0), axis=0).astype(np.float32)
    return inter, chosen_paths


def copy_midi(src_midi: Path, dst_midi: Path):
    ensure_dir(dst_midi.parent)
    shutil.copy2(src_midi, dst_midi)


def write_audio(path: Path, y: np.ndarray, sr: int):
    ensure_dir(path.parent)
    sf.write(str(path), y, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maestro_root", type=str, required=True)
    parser.add_argument("--musdb_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=1377)
    args = parser.parse_args()

    set_seed(args.seed)

    maestro_root = Path(args.maestro_root)
    musdb_root = Path(args.musdb_root)
    out_root = Path(args.out_root)

    meta_path = out_root / "metadata.csv"
    ensure_dir(out_root)

    maestro_pairs = find_maestro_pairs(maestro_root)
    musdb_index = index_musdb_stems(musdb_root)
    n_samples = len(maestro_pairs)

    print(f"Generating {n_samples} samples...")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "mode",
                "progress",
                "target_snr_db",
                "maestro_audio",
                "maestro_midi",
                "musdb_stems",
                "out_audio",
                "out_midi",
            ],
        )
        writer.writeheader()

        for i in tqdm(range(n_samples)):
            mode = choose_mode(i, n_samples)
            progress = i / max(1, n_samples - 1)

            ma_audio_path, ma_midi_path = random.choice(maestro_pairs)
            clean = load_audio(ma_audio_path, args.sr)
            target_len = len(clean)

            inter, used_stems = mix_interferers(
                mode=mode,
                musdb_index=musdb_index,
                target_len=target_len,
                sr=args.sr,
            )

            snr_db = np.nan
            if mode != "clean":
                snr_db = sample_snr_db(progress)
                inter = scale_to_snr(clean, inter, snr_db)

            mix = clean + inter

            out_audio_rel = ma_audio_path.relative_to(maestro_root).with_suffix(".wav")
            out_midi_rel = ma_midi_path.relative_to(maestro_root)

            out_audio_path = out_root / out_audio_rel
            out_midi_path = out_root / out_midi_rel
            sample_id = str(out_audio_rel.with_suffix(""))

            write_audio(out_audio_path, mix, args.sr)
            copy_midi(ma_midi_path, out_midi_path)

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "mode": mode,
                    "progress": f"{progress:.6f}",
                    "target_snr_db": "" if mode == "clean" else f"{snr_db:.3f}",
                    "maestro_audio": str(ma_audio_path),
                    "maestro_midi": str(ma_midi_path),
                    "musdb_stems": ";".join(str(p) for p in used_stems),
                    "out_audio": str(out_audio_path),
                    "out_midi": str(out_midi_path),
                }
            )

    print("\nProcessing complete")
    print(f"dataset: {out_root}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
