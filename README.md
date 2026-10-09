# go-silerovad

[Silero VAD](https://github.com/snakers4/silero-vad) v6 voice activity detection in **pure Go**.
No CGO, onnxruntime, shared libraries, or model files are needed at runtime: the original
network weights are embedded in the binary, and inference is implemented directly from
the ONNX graph. Builds with `CGO_ENABLED=0` for any architecture supported by Go.

Results differ from onnxruntime running the original `silero_vad.onnx` by at most 1e-4
(verified on real speech by `TestModelMatchesOnnxRuntime`).

## Usage

Module: `github.com/vithnilica/go-silerovad`, package `silerovad`, Go 1.26 or later.
No dependencies outside the standard library.

```bash
go get github.com/vithnilica/go-silerovad
```

### Speech segments in a recording

```go
import silerovad "github.com/vithnilica/go-silerovad"

// samples: mono float32 in the range −1…1, any sample rate from 8 to 192 kHz.
segments, err := silerovad.Segments(samples, 44100, silerovad.SegmenterOptions{})
if err != nil {
    log.Fatal(err)
}
for _, s := range segments {
    fmt.Printf("%.2f–%.2f s\n", s.Start.Seconds(), s.End.Seconds())
}
```

### Live streaming

`Stream` accepts audio in chunks of any size, resamples it to 16 kHz, and returns a
speech probability for each complete 32 ms chunk. `Segmenter` turns these probabilities
into speech start and end events. When the input ends, `Flush` processes the remaining
audio, padding the last chunk with silence.

```go
stream, err := silerovad.NewStream(48000)
if err != nil {
    log.Fatal(err)
}
segmenter := silerovad.NewSegmenter(silerovad.SegmenterOptions{})
for frame := range microphone { // PCM16 little-endian mono
    for _, p := range stream.PushPCM16(frame) {
        if event, ok := segmenter.Next(p); ok {
            fmt.Println(event) // speech start 1.234s / speech end 2.5s
        }
    }
}
```

### Direct model access

`Detector` is the lowest-level API: consecutive chunks of exactly 512 samples at 16 kHz.

```go
d, _ := silerovad.NewDetector()
p, err := d.Process(chunk) // len(chunk) == silerovad.ChunkSize
```

Neither `Detector` nor `Stream` is safe for concurrent use; create a separate instance
for each audio stream. Weights are loaded once and shared read-only by all instances
(about 1.2 MB of shared memory, with around 11 kB of state per detector).

### Examples

- [examples/wav](examples/wav/main.go) prints speech segments in a WAV file (PCM16, mono or stereo):
  `go run ./examples/wav recording.wav`
- [examples/live](examples/live/main.go) reports speech from microphone audio piped through standard input:
  `arecord -q -f S16_LE -r 16000 -c 1 | go run ./examples/live`,
  use `-meter` to display the probability for each chunk.
- [example_test.go](example_test.go) contains runnable examples for `go doc`.

### Segmentation options

Zero-valued `SegmenterOptions` use the defaults from the upstream `VADIterator`:

| Option | Default | Meaning |
| --- | --- | --- |
| `Threshold` | 0.5 | speech start threshold |
| `NegThreshold` | `Threshold` − 0.15 | speech end threshold; the state stays unchanged between thresholds (the offset is fixed in Python) |
| `MinSilence` | 100 ms | silence duration required to end speech |
| `SpeechPad` | 30 ms | padding added to both ends of each segment |
| `MinSpeech` | 250 ms | `Segments` discards segments shorter than this before padding |

`Segments` is a simplified equivalent of `get_speech_timestamps`: like the original, it keeps
padded segments within the recording and splits gaps shorter than twice `SpeechPad`
between adjacent segments, but it does not split long segments (`max_speech_duration_s`).

## How it works

Silero VAD v6 is a small network (about 310,000 weights). This package embeds its
16 kHz branch from `silero_vad.onnx` release **v6.2.1** (the same file as in v6.2
through v6.2.3, SHA-256 `1a153a22…88e3`). One forward pass ([model.go](model.go)):

1. **Input**: 512 new samples and 64 context samples from the end of the previous
   chunk (as in Python's `OnnxWrapper`), with 64 samples of reflection padding on the right.
2. **STFT** as a convolution with a fixed 258 × 256 basis and stride 128. The basis
   matches a DFT with a periodic Hann window (verified within 8e-8). The four windows
   produce magnitudes for 129 frequency bins.
3. **Encoder**: four Conv1d layers with kernel size 3 and ReLU (129 → 128 → 64 → 64 → 128
   channels, strides 1, 2, 2, 1), with time dimensions 4 → 4 → 2 → 1 → 1.
4. **Decoder**: `LSTMCell` with 128 units (PyTorch gate order i, f, g, o),
   ReLU, 1 × 1 Conv1d, and sigmoid. The LSTM state carries over between chunks.

The ONNX graph contains several branches (`If`) solely to handle tensor shapes and
sample rates; the Go implementation follows the single path used at 16 kHz with a
batch size of 1. ONNX reorders LSTM weights to i, o, f, c; this library uses the
original PyTorch order directly from the `decoder.rnn.*` constants.

**Resampling** ([resample.go](resample.go)) uses a polyphase FIR filter: a sinc with
a Blackman window and a cutoff at 93.75% of the lower Nyquist frequency (7.5 kHz when
downsampling to 16 kHz). The sample rate ratio is reduced, giving 160 phases for
44.1 kHz input. The 300–3400 Hz speech band is preserved within 3%, while 10 kHz is
attenuated to below 1%. The network also has an 8 kHz branch; it is not embedded,
so 8 kHz input is resampled to 16 kHz.

## Validation and performance

```bash
go test -race ./...
go test -run x -bench . ./...
```

- **Agreement with onnxruntime**: 187 chunks (6 s) of real speech from Silero's test
  recording, with an error below 1e-4. Reference probabilities were computed by
  onnxruntime 1.31 using the original model; see [testdata/speech-16k.json](testdata/speech-16k.json).
- **The full 24 kHz pipeline** (resampling and inference) agrees with the reference
  classification for 97% of chunks, with a mean absolute error below 0.03. The remaining
  difference comes from the library's resampling versus the exact FFT resampling used
  to generate the test data.
- The closest test probability is 0.006 away from the 0.5 threshold, so example outputs
  are also identical on arm64, where Go uses fused multiply-add (FMA).

On an Intel Xeon Gold 6230R (2.1 GHz, one core), a 32 ms chunk takes about **0.95 ms**,
or roughly 3% of a core per real-time audio stream (34× faster than real time).
A 60-second recording takes about 2 s to process.

## Regenerating weights and test data

[tools/export_weights.py](tools/export_weights.py) downloads the model from a pinned
tag, verifies its SHA-256, exports the weights to `silero_vad.bin`, and generates
test data and reference outputs from `tests/data/test.wav` in the same release.
The output is bit-for-bit reproducible.

```bash
pip install onnx onnxruntime numpy
python3 tools/export_weights.py
```

The `silero_vad.bin` format consists of the `SILERO6\0` header followed by little-endian
float32 arrays in a fixed order, each prefixed by its element count. The script and
[model.go](model.go) use the same order.

## References

- Silero Team: [Silero VAD](https://github.com/snakers4/silero-vad), release
  [v6.2.1](https://github.com/snakers4/silero-vad/releases/tag/v6.2.1).
  Source of the model, architecture, `VADIterator` defaults, and test recording.
  The 64-sample context follows `OnnxWrapper` in `src/silero_vad/utils_vad.py`.
- ONNX: [operator specifications](https://onnx.ai/onnx/operators/) for Conv, LSTM,
  Pad (reflect), Slice, and ReduceMean, used to implement the graph.
- PyTorch: [LSTMCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)
  (gate order and equations) and [reflection padding](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReflectionPad1d.html).
- Resampling: windowed sinc and the Blackman window, as described by A. V. Oppenheim
  and R. W. Schafer in *Discrete-Time Signal Processing*, in the sections on FIR filter
  design using windows and sample rate conversion by a rational factor.

Related projects: [zserge/govad](https://github.com/zserge/govad) (pure Go,
Silero VAD v5 with half-precision weights, 16 kHz only) and
[346/silero-vad-go](https://github.com/346/silero-vad-go) (CGO and onnxruntime).
go-silerovad does not reuse their code; it provides the v6 model at full precision,
resampling from arbitrary input sample rates within the supported range, and segmentation.

### Model citation

The Silero VAD authors request the following citation when using the model:

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

