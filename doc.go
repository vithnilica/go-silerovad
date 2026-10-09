// Package silerovad implements Silero VAD v6 voice activity detection in pure Go.
//
// No CGO, onnxruntime, or model files are needed: the original network weights are
// embedded in the binary, and inference is implemented directly from the ONNX graph.
// Results differ from onnxruntime running the original model by at most 1e-4
// (see TestModelMatchesOnnxRuntime).
//
// The API has three levels:
//
//   - [Detector] processes chunks of 512 samples at 16 kHz and returns a speech probability.
//   - [Stream] accepts audio at any supported sample rate, resamples it to 16 kHz,
//     and returns probabilities for complete chunks (32 ms).
//   - [Segmenter] and [Segments] turn probabilities into speech segments with start
//     and end times, like the upstream VADIterator and get_speech_timestamps.
//
// The model, weights, and test recording come from the Silero VAD project
// (https://github.com/snakers4/silero-vad, MIT, © Silero Team); see README.md
// and THIRD_PARTY_NOTICES.md for details.
package silerovad
