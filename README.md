# Voxtral Benchmark

Benchmarks latency and WER for Voxtral Mini and Voxtral Realtime on LibriSpeech.

## Setup

```
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```
python benchmark.py -n 10
python benchmark.py -n 20 --models mini
python benchmark.py -n 20 --models realtime
python benchmark.py -n 50 --output results.json
```
