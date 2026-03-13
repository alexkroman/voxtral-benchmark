#!/usr/bin/env python3
"""Benchmark end-of-turn latency and WER for Voxtral Mini and Voxtral Realtime on LibriSpeech."""

import argparse
import time
import json
import tempfile
import os

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from jiwer import wer


def get_default_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device():
    """Synchronize GPU/MPS for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_cache():
    """Free device memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_librispeech(num_samples: int, split: str = "test.clean"):
    """Stream LibriSpeech and collect the requested number of samples."""
    ds = load_dataset("openslr/librispeech_asr", split=split, streaming=True)
    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        samples.append(sample)
        print(f"\r  Loaded {i+1}/{num_samples} samples", end="", flush=True)
    print()
    return samples


def save_audio_to_tempfile(audio_array: np.ndarray, sample_rate: int) -> str:
    """Write audio array to a temporary WAV file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio_array, sample_rate)
    return path


def benchmark_voxtral_mini(samples, device: str):
    """Benchmark Voxtral-Mini-3B transcription latency and WER."""
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    model_id = "mistralai/Voxtral-Mini-3B-2507"
    print(f"\n{'='*60}")
    print(f"Loading {model_id}...")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    results = []
    for i, sample in enumerate(samples):
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        duration_s = len(audio_array) / sr
        reference = sample["text"]

        tmp_path = save_audio_to_tempfile(audio_array, sr)
        try:
            inputs = processor.apply_transcription_request(
                language="en",
                audio=tmp_path,
                model_id=model_id,
            )
            inputs = inputs.to(model.device, dtype=torch.bfloat16)

            sync_device()

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            sync_device()
            t1 = time.perf_counter()

            latency = t1 - t0
            hypothesis = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0]
        finally:
            os.unlink(tmp_path)

        sample_wer = wer(reference.lower(), hypothesis.lower())

        results.append({
            "sample_idx": i,
            "audio_duration_s": round(duration_s, 2),
            "latency_s": round(latency, 4),
            "rtf": round(latency / duration_s, 4) if duration_s > 0 else None,
            "wer": round(sample_wer, 4),
            "reference": reference,
            "hypothesis": hypothesis,
        })
        print(
            f"  [{i+1}/{len(samples)}] "
            f"audio={duration_s:.1f}s  latency={latency:.2f}s  "
            f"RTF={latency/duration_s:.3f}  WER={sample_wer:.2%}"
        )

    del model, processor
    empty_cache()
    return results


def benchmark_voxtral_realtime(samples, device: str):
    """Benchmark Voxtral-Realtime transcription latency and WER."""
    from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor

    model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    print(f"\n{'='*60}")
    print(f"Loading {model_id}...")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    results = []
    for i, sample in enumerate(samples):
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        duration_s = len(audio_array) / sr
        reference = sample["text"]

        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)

        sync_device()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        sync_device()
        t1 = time.perf_counter()

        latency = t1 - t0
        hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        sample_wer = wer(reference.lower(), hypothesis.lower())

        results.append({
            "sample_idx": i,
            "audio_duration_s": round(duration_s, 2),
            "latency_s": round(latency, 4),
            "rtf": round(latency / duration_s, 4) if duration_s > 0 else None,
            "wer": round(sample_wer, 4),
            "reference": reference,
            "hypothesis": hypothesis,
        })
        print(
            f"  [{i+1}/{len(samples)}] "
            f"audio={duration_s:.1f}s  latency={latency:.2f}s  "
            f"RTF={latency/duration_s:.3f}  WER={sample_wer:.2%}"
        )

    del model, processor
    empty_cache()
    return results


def print_summary(name: str, results: list):
    """Print aggregate statistics for a model's results."""
    if not results:
        return
    latencies = [r["latency_s"] for r in results]
    rtfs = [r["rtf"] for r in results if r["rtf"] is not None]
    durations = [r["audio_duration_s"] for r in results]
    wers = [r["wer"] for r in results]

    # Compute corpus-level WER from all references/hypotheses
    all_refs = [r["reference"].lower() for r in results]
    all_hyps = [r["hypothesis"].lower() for r in results]
    corpus_wer = wer(all_refs, all_hyps)

    print(f"\n{'='*60}")
    print(f"  {name} — Summary ({len(results)} samples)")
    print(f"{'='*60}")
    print(f"  Audio duration : mean={np.mean(durations):.1f}s  total={np.sum(durations):.1f}s")
    print(f"  Latency        : mean={np.mean(latencies):.2f}s  median={np.median(latencies):.2f}s  "
          f"std={np.std(latencies):.2f}s")
    print(f"                   min={np.min(latencies):.2f}s  max={np.max(latencies):.2f}s")
    if rtfs:
        print(f"  RTF            : mean={np.mean(rtfs):.4f}  median={np.median(rtfs):.4f}")
    print(f"  WER (corpus)   : {corpus_wer:.2%}")
    print(f"  WER (per-sample): mean={np.mean(wers):.2%}  median={np.median(wers):.2%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Voxtral models on LibriSpeech")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=10,
        help="Number of samples to benchmark (default: 10)",
    )
    parser.add_argument(
        "--split", type=str, default="test.clean",
        choices=["test.clean", "test.other", "validation.clean", "validation.other"],
        help="LibriSpeech split (default: test.clean)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=["mini", "realtime", "both"], default=["both"],
        help="Which models to benchmark (default: both)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for model loading (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (optional)",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = get_default_device()
    print(f"Using device: {args.device}")

    models_to_run = set()
    for m in args.models:
        if m == "both":
            models_to_run.update(["mini", "realtime"])
        else:
            models_to_run.add(m)

    print(f"Streaming LibriSpeech ({args.split}, {args.num_samples} samples)...")
    samples = load_librispeech(args.num_samples, args.split)
    print(f"Loaded {len(samples)} samples.")

    all_results = {}

    if "mini" in models_to_run:
        results = benchmark_voxtral_mini(samples, args.device)
        all_results["voxtral_mini"] = results
        print_summary("Voxtral Mini (3B)", results)

    if "realtime" in models_to_run:
        results = benchmark_voxtral_realtime(samples, args.device)
        all_results["voxtral_realtime"] = results
        print_summary("Voxtral Realtime (4B)", results)

    if "voxtral_mini" in all_results and "voxtral_realtime" in all_results:
        mini_lat = np.mean([r["latency_s"] for r in all_results["voxtral_mini"]])
        rt_lat = np.mean([r["latency_s"] for r in all_results["voxtral_realtime"]])
        mini_wer = wer(
            [r["reference"].lower() for r in all_results["voxtral_mini"]],
            [r["hypothesis"].lower() for r in all_results["voxtral_mini"]],
        )
        rt_wer = wer(
            [r["reference"].lower() for r in all_results["voxtral_realtime"]],
            [r["hypothesis"].lower() for r in all_results["voxtral_realtime"]],
        )
        print(f"{'='*60}")
        print(f"  Comparison")
        print(f"{'='*60}")
        print(f"  Voxtral Mini     : latency={mini_lat:.2f}s  WER={mini_wer:.2%}")
        print(f"  Voxtral Realtime : latency={rt_lat:.2f}s  WER={rt_wer:.2%}")
        faster = "Realtime" if rt_lat < mini_lat else "Mini"
        ratio = max(mini_lat, rt_lat) / min(mini_lat, rt_lat)
        print(f"  → {faster} is {ratio:.2f}x faster")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
