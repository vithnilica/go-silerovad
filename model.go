package silerovad

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"math"
	"sync"
)

const (
	// SampleRate is the sample rate used by the network.
	SampleRate = 16000
	// ChunkSize is the number of samples per inference step (32 ms at 16 kHz).
	ChunkSize = 512
	// ModelVersion is the Silero VAD release that provides the embedded weights.
	ModelVersion = "v6.2.1"
	// ModelSHA256 is the checksum of the original silero_vad.onnx from this release.
	ModelSHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)

// Weights for the model's 16 kHz branch, in the format used by tools/export_weights.py.
//
//go:embed silero_vad.bin
var weightsFile []byte

const (
	contextSize = 64  // samples from the end of the previous chunk
	padSize     = 64  // reflection padding on the right before STFT
	hidden      = 128 // LSTM state size
	frames      = 4   // (64 + 512 + 64 − 256) / 128 + 1 STFT windows
	bins        = 129 // one-sided spectrum of a 256-point STFT
)

// ErrInvalidChunk is returned by Detector.Process when a chunk's length is not ChunkSize.
var ErrInvalidChunk = errors.New("silerovad: chunk must contain 512 samples")

type weights struct {
	basis               []float32 // 258 × 256: real and imaginary rows of the windowed STFT basis
	enc0w, enc1w, enc2w []float32
	enc3w               []float32
	enc0b, enc1b, enc2b []float32
	enc3b               []float32
	wih, whh, bih, bhh  []float32
	decoderW            []float32
	decoderB            float32
}

// Weights are loaded once and shared by all detectors; they are immutable after loading.
var loadWeights = sync.OnceValues(func() (*weights, error) {
	invalid := errors.New("silerovad: corrupt weights file")
	data := weightsFile
	if len(data) < 8 || string(data[:8]) != "SILERO6\x00" {
		return nil, invalid
	}
	data = data[8:]
	next := func(size int) []float32 {
		if data == nil || len(data) < 4+size*4 || int(binary.LittleEndian.Uint32(data)) != size {
			data = nil
			return nil
		}
		out := make([]float32, size)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4+i*4:]))
		}
		data = data[4+size*4:]
		return out
	}
	w := &weights{}
	// The order matches WEIGHTS in tools/export_weights.py.
	w.basis = next(258 * 256)
	w.enc0w, w.enc0b = next(128*129*3), next(128)
	w.enc1w, w.enc1b = next(64*128*3), next(64)
	w.enc2w, w.enc2b = next(64*64*3), next(64)
	w.enc3w, w.enc3b = next(128*64*3), next(128)
	w.wih, w.whh = next(4*hidden*hidden), next(4*hidden*hidden)
	w.bih, w.bhh = next(4*hidden), next(4*hidden)
	w.decoderW = next(hidden)
	bias := next(1)
	if data == nil || len(data) != 0 {
		return nil, invalid
	}
	w.decoderB = bias[0]
	return w, nil
})

// Detector holds the network state for one audio stream. It is not safe for
// concurrent use; create separate detectors for separate streams (weights are shared).
type Detector struct {
	w       *weights
	context [contextSize]float32
	h, c    [hidden]float32
	padded  [contextSize + ChunkSize + padSize]float32
	spec    [bins * frames]float32
	a0      [128 * 4]float32
	a1      [64 * 2]float32
	a2      [64 * 1]float32
	a3      [128 * 1]float32
	gates   [4 * hidden]float32
}

// NewDetector creates a detector with zeroed state.
func NewDetector() (*Detector, error) {
	w, err := loadWeights()
	if err != nil {
		return nil, err
	}
	return &Detector{w: w}, nil
}

// Process evaluates 512 samples at 16 kHz in the range −1…1 and returns a speech
// probability in the range 0…1. Chunks must be consecutive; the network retains
// context and LSTM state from previous chunks.
func (d *Detector) Process(chunk []float32) (float32, error) {
	if len(chunk) != ChunkSize {
		return 0, ErrInvalidChunk
	}
	return d.process(chunk), nil
}

// Reset clears the network state, for example before processing a different recording.
func (d *Detector) Reset() {
	d.context, d.h, d.c = [contextSize]float32{}, [hidden]float32{}, [hidden]float32{}
}

func (d *Detector) process(chunk []float32) float32 {
	w := d.w
	// Network input: 64 context samples, the chunk, and reflection padding (torch reflect pad).
	x := d.padded[:]
	copy(x, d.context[:])
	copy(x[contextSize:], chunk)
	n := contextSize + ChunkSize
	for i := 0; i < padSize; i++ {
		x[n+i] = x[n-2-i]
	}
	copy(d.context[:], x[n-contextSize:n])

	// STFT as a convolution with kernel size 256 and stride 128; magnitude from real and imaginary parts.
	for f := 0; f < bins; f++ {
		re, im := w.basis[f*256:(f+1)*256], w.basis[(f+bins)*256:(f+bins+1)*256]
		for t := 0; t < frames; t++ {
			window := x[t*128 : t*128+256]
			var sr, si float32
			for k, v := range window {
				sr += re[k] * v
				si += im[k] * v
			}
			d.spec[f*frames+t] = float32(math.Sqrt(float64(sr*sr + si*si)))
		}
	}
	// Encoder: four Conv1d layers (kernel size 3, padding 1) with ReLU; time dimensions 4 → 4 → 2 → 1 → 1.
	conv(d.spec[:], bins, frames, w.enc0w, w.enc0b, 128, 1, d.a0[:])
	conv(d.a0[:], 128, 4, w.enc1w, w.enc1b, 64, 2, d.a1[:])
	conv(d.a1[:], 64, 2, w.enc2w, w.enc2b, 64, 2, d.a2[:])
	conv(d.a2[:], 64, 1, w.enc3w, w.enc3b, 128, 1, d.a3[:])

	// Decoder: LSTMCell in PyTorch gate order (input, forget, candidate, output),
	// then ReLU, 1×1 Conv1d, and sigmoid. Averaging over time is a no-op for a single frame.
	for g := 0; g < 4*hidden; g++ {
		sum := w.bih[g] + w.bhh[g]
		rowI, rowH := w.wih[g*hidden:(g+1)*hidden], w.whh[g*hidden:(g+1)*hidden]
		for k := 0; k < hidden; k++ {
			sum += rowI[k]*d.a3[k] + rowH[k]*d.h[k]
		}
		d.gates[g] = sum
	}
	out := w.decoderB
	for k := 0; k < hidden; k++ {
		i := sigmoid(d.gates[k])
		f := sigmoid(d.gates[hidden+k])
		g := float32(math.Tanh(float64(d.gates[2*hidden+k])))
		o := sigmoid(d.gates[3*hidden+k])
		d.c[k] = f*d.c[k] + i*g
		d.h[k] = o * float32(math.Tanh(float64(d.c[k])))
		if d.h[k] > 0 {
			out += w.decoderW[k] * d.h[k]
		}
	}
	return sigmoid(out)
}

// conv applies Conv1d with kernel size 3, padding 1, and ReLU; input and output use [channel][time] layout.
func conv(in []float32, cin, tin int, weight, bias []float32, cout, stride int, out []float32) {
	tout := (tin+2-3)/stride + 1
	for o := 0; o < cout; o++ {
		for t := 0; t < tout; t++ {
			sum := bias[o]
			for c := 0; c < cin; c++ {
				kernel := weight[(o*cin+c)*3 : (o*cin+c)*3+3]
				row := in[c*tin : (c+1)*tin]
				for k := 0; k < 3; k++ {
					if pos := t*stride + k - 1; pos >= 0 && pos < tin {
						sum += kernel[k] * row[pos]
					}
				}
			}
			out[o*tout+t] = max(sum, 0)
		}
	}
}

func sigmoid(v float32) float32 { return float32(1 / (1 + math.Exp(-float64(v)))) }
