#!/usr/bin/env python3
"""Reproducible export of Silero VAD weights for go-silerovad.

Downloads the pinned model, verifies its SHA-256, saves the 16 kHz branch weights
to silero_vad.bin, and generates reference probabilities with onnxruntime from
Silero's test recording. Go tests use these to verify that the Go implementation
produces the same results.

Run from the repository root:

    pip install onnx onnxruntime numpy
    python3 tools/export_weights.py

The silero_vad.bin format consists of the "SILERO6\0" header followed by little-endian
float32 arrays in WEIGHTS order, each prefixed by its element count (uint32 LE).
If the weights' SHA-256 is unchanged after export, the output is bit-for-bit identical.
"""
import hashlib
import json
import struct
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

TAG = "v6.2.1"
BASE = f"https://raw.githubusercontent.com/snakers4/silero-vad/{TAG}"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT

# The order and shapes must match the loader in model.go.
WEIGHTS = [
    ("stft.forward_basis_buffer", (258, 1, 256)),
    ("encoder.0.reparam_conv.weight", (128, 129, 3)),
    ("encoder.0.reparam_conv.bias", (128,)),
    ("encoder.1.reparam_conv.weight", (64, 128, 3)),
    ("encoder.1.reparam_conv.bias", (64,)),
    ("encoder.2.reparam_conv.weight", (64, 64, 3)),
    ("encoder.2.reparam_conv.bias", (64,)),
    ("encoder.3.reparam_conv.weight", (128, 64, 3)),
    ("encoder.3.reparam_conv.bias", (128,)),
    ("decoder.rnn.weight_ih", (512, 128)),
    ("decoder.rnn.weight_hh", (512, 128)),
    ("decoder.rnn.bias_ih", (512,)),
    ("decoder.rnn.bias_hh", (512,)),
    ("decoder.decoder.2.weight", (1, 128, 1)),
    ("decoder.decoder.2.bias", (1,)),
]


def fetch(path):
    with urllib.request.urlopen(f"{BASE}/{path}") as response:
        return response.read()


def weights_16k(model):
    # The model branches by sample rate; then_branch handles 16 kHz.
    branch = next(a.g for n in model.graph.node if n.op_type == "If" for a in n.attribute if a.name == "then_branch")
    found = {}
    for node in branch.node:
        if node.op_type == "Constant":
            name = node.output[0].replace("If_0_then_branch__Inline_0__", "")
            found[name] = numpy_helper.to_array(node.attribute[0].t)
    return found


def probabilities(session, audio):
    # Follow Silero's OnnxWrapper: 512 samples plus 64 context samples.
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    out = []
    for start in range(0, len(audio) - 511, 512):
        chunk = np.concatenate([context, audio[None, start:start + 512]], axis=1)
        prob, state = session.run(None, {"input": chunk, "state": state, "sr": np.array(16000, dtype=np.int64)})
        context = chunk[:, -64:]
        out.append(float(prob[0, 0]))
    return out


def resample_16_to_24(audio):
    # Exact band-limited resampling via FFT; used only for the 24 kHz test input.
    spectrum = np.fft.rfft(audio)
    n = len(audio) * 3 // 2
    padded = np.zeros(n // 2 + 1, dtype=complex)
    padded[: len(spectrum)] = spectrum
    return np.fft.irfft(padded, n) * 1.5


def main():
    raw = fetch("src/silero_vad/data/silero_vad.onnx")
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        sys.exit("Unexpected SHA-256 for the Silero VAD model")
    model = onnx.load_from_string(raw)
    found = weights_16k(model)
    with open(OUT / "silero_vad.bin", "wb") as f:
        f.write(b"SILERO6\0")
        for name, shape in WEIGHTS:
            array = found[name].astype("<f4")
            assert array.shape == shape, (name, array.shape)
            f.write(struct.pack("<I", array.size))
            f.write(array.tobytes())
    (OUT / "LICENSE-silero").write_bytes(fetch("LICENSE"))
    print("silero_vad.bin", hashlib.sha256((OUT / "silero_vad.bin").read_bytes()).hexdigest())

    wav_path = OUT / "testdata"
    wav_path.mkdir(exist_ok=True)
    tmp = wav_path / "test.wav"
    tmp.write_bytes(fetch("tests/data/test.wav"))
    with wave.open(str(tmp)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    tmp.unlink()
    session = onnxruntime.InferenceSession(raw, providers=["CPUExecutionProvider"])
    whole = probabilities(session, pcm.astype(np.float32) / 32768)
    # Select the 6-second excerpt with the most speech transitions, including starts and ends.
    window = 6 * 16000 // 512
    speech = np.array(whole) > 0.5
    best = max(range(len(whole) - window), key=lambda i: int(np.sum(speech[i + 1:i + window] != speech[i:i + window - 1])))
    excerpt = pcm[best * 512:(best + window) * 512]
    (wav_path / "speech-16k.pcm").write_bytes(excerpt.astype("<i2").tobytes())
    up = np.clip(np.round(resample_16_to_24(excerpt.astype(np.float64))), -32768, 32767).astype("<i2")
    (wav_path / "speech-24k.pcm").write_bytes(up.tobytes())
    golden = probabilities(session, excerpt.astype(np.float32) / 32768)
    (wav_path / "speech-16k.json").write_text(json.dumps({"source": f"silero-vad {TAG} tests/data/test.wav", "offset_chunks": best, "probabilities": [round(p, 6) for p in golden]}) + "\n")
    print("excerpt starts at", best * 512 / 16000, "s, speech in", sum(p > 0.5 for p in golden), "of", len(golden), "chunks")


if __name__ == "__main__":
    main()
