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

Requirements:
  pip install librosa soundfile numpy pandas pyloudnorm tqdm
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
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}
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
    """
    Tries to pair MAESTRO audio and midi by stem-relative path.
    If maestro-v*.csv exists, uses it.
    Else fallback: recursively match by filename stem.
    """
    csv_candidates = sorted(maestro_root.glob("maestro-v*.csv"))
    pairs: List[Tuple[Path, Path]] = []

    if csv_candidates:
        # Use latest csv by lexical sort
        maestro_csv = csv_candidates[-1]
        df = pd.read_csv(maestro_csv)
        # Expected columns in MAESTRO: audio_filename, midi_filename
        if "audio_filename" in df.columns and "midi_filename" in df.columns:
            for _, row in df.iterrows():
                a = maestro_root / str(row["audio_filename"])
                m = maestro_root / str(row["midi_filename"])
                if a.exists() and m.exists():
                    pairs.append((a, m))
        if pairs:
            return pairs

    # fallback
    audios = [p for p in maestro_root.rglob("*") if p.is_file() and is_audio_file(p)]
    midi_map: Dict[str, Path] = {}
    for p in maestro_root.rglob("*"):
        if p.is_file() and is_midi_file(p):
            midi_map[p.stem] = p

    for a in audios:
        if a.stem in midi_map:
            pairs.append((a, midi_map[a.stem]))

    # de-dup by audio path
    seen = set()
    uniq = []
    for a, m in pairs:
        k = str(a.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append((a, m))
    return uniq


def index_musdb_stems(musdb_root: Path) -> Dict[str, List[Path]]:
    """
    Index MUSDB stems by inferred stem type from filename:
      vocals, drums, bass
    """
    stems = {"vocals": [], "drums": [], "bass": []}
    for p in musdb_root.rglob("*"):
        if not (p.is_file() and is_audio_file(p)):
            continue
        name = p.name.lower()
        # common musdb stem names: vocals.wav, drums.wav, bass.wav
        if "vocals" in name:
            stems["vocals"].append(p)
        elif "drums" in name:
            stems["drums"].append(p)
        elif "bass" in name:
            stems["bass"].append(p)
    return stems


def load_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def peak_normalize(y: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak_lin = 10 ** (peak_db / 20.0)
    m = np.max(np.abs(y)) + 1e-12
    return (y / m) * peak_lin


def loudness_normalize(
    y: np.ndarray, sr: int, meter: pyln.Meter, target_lufs: float = -14.0
) -> np.ndarray:
    if len(y) < sr // 2:
        return y
    loud = meter.integrated_loudness(y)
    y_n = pyln.normalize.loudness(y, loud, target_lufs)
    return y_n.astype(np.float32)


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
    """
    progress in [0,1]
    curriculum:
      <=0.3: uniform [20, 5]
      >0.3 : uniform [20, -5]
    """
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
    stem_index: Dict[str, List[Path]],
    target_len: int,
    sr: int,
    meter: pyln.Meter,
    target_lufs: float,
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
        pool = stem_index.get(pool_name, [])
        if not pool:
            return None
        p = random.choice(pool)
        y = load_mono(p, sr)
        y = tile_or_crop(y, target_len)
        y = loudness_normalize(y, sr, meter, target_lufs=target_lufs)
        chosen_paths.append(p)
        return y

    if mode == "clean":
        return np.zeros(target_len, dtype=np.float32), chosen_paths

    if mode == "noise_only":
        # use one or more of drums/bass
        candidates = [k for k in ["drums", "bass"] if len(stem_index.get(k, [])) > 0]
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
        candidates = [k for k in ["drums", "bass"] if len(stem_index.get(k, [])) > 0]
        if candidates:
            # one or more non-vocal stems
            k = random.randint(1, len(candidates))
            picked_types = random.sample(candidates, k=k)
            for t in picked_types:
                yn = pick_and_prepare(t)
                if yn is not None:
                    parts.append(yn)

    if not parts:
        return np.zeros(target_len, dtype=np.float32), chosen_paths

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
    parser.add_argument("--target_lufs", type=float, default=-14.0)
    parser.add_argument(
        "--n_samples", type=int, default=0, help="0 = use all MAESTRO pairs once"
    )
    parser.add_argument("--seed", type=int, default=1377)
    parser.add_argument("--peak_db", type=float, default=-1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    maestro_root = Path(args.maestro_root)
    musdb_root = Path(args.musdb_root)
    out_root = Path(args.out_root)

    out_audio = out_root / "audio"
    out_midi = out_root / "midi"
    meta_path = out_root / "metadata.csv"
    ensure_dir(out_root)
    ensure_dir(out_audio)
    ensure_dir(out_midi)

    print("Indexing MAESTRO...")
    maestro_pairs = find_maestro_pairs(maestro_root)
    if not maestro_pairs:
        raise RuntimeError("No MAESTRO (audio, midi) pairs found.")

    print("Indexing MUSDB stems...")
    stem_index = index_musdb_stems(musdb_root)
    for k in ["vocals", "drums", "bass"]:
        print(f"  {k}: {len(stem_index[k])}")
    if len(stem_index["vocals"]) == 0:
        raise RuntimeError("No vocals stems found in MUSDB root.")
    if len(stem_index["drums"]) == 0 and len(stem_index["bass"]) == 0:
        raise RuntimeError("No drums/bass stems found in MUSDB root.")

    n = args.n_samples if args.n_samples > 0 else len(maestro_pairs)
    meter = pyln.Meter(args.sr)

    # If n > len(maestro_pairs), sample with replacement.
    def get_pair(i: int) -> Tuple[Path, Path]:
        if i < len(maestro_pairs):
            return maestro_pairs[i]
        return random.choice(maestro_pairs)

    print(f"Generating {n} samples...")
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

        for i in tqdm(range(n)):
            mode = choose_mode(i, n)
            progress = i / max(1, n - 1)

            ma_audio_path, ma_midi_path = get_pair(i)
            clean = load_mono(ma_audio_path, args.sr)
            clean = loudness_normalize(
                clean, args.sr, meter, target_lufs=args.target_lufs
            )

            target_len = len(clean)

            inter, used_stems = mix_interferers(
                mode=mode,
                stem_index=stem_index,
                target_len=target_len,
                sr=args.sr,
                meter=meter,
                target_lufs=args.target_lufs,
            )

            snr_db = np.nan
            if mode != "clean":
                snr_db = sample_snr_db(progress)
                inter = scale_to_snr(clean, inter, snr_db)

            mix = clean + inter
            mix = peak_normalize(mix, peak_db=args.peak_db).astype(np.float32)

            sample_id = f"{i:07d}"
            out_audio_path = out_audio / f"{sample_id}.wav"
            out_midi_path = out_midi / f"{sample_id}{ma_midi_path.suffix.lower()}"

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

    print(f"Done. Wrote: {out_root}")
    print(f"- audio: {out_audio}")
    print(f"- midi : {out_midi}")
    print(f"- meta : {meta_path}")


if __name__ == "__main__":
    main()
