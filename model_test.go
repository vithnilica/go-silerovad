package silerovad

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"
)

func readPCM(t testing.TB, name string) []float32 {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/2)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) / 32768
	}
	return out
}

// golden returns the probabilities computed by onnxruntime using the original model.
func golden(t testing.TB) []float64 {
	t.Helper()
	raw, err := os.ReadFile("testdata/speech-16k.json")
	if err != nil {
		t.Fatal(err)
	}
	var v struct {
		Probabilities []float64 `json:"probabilities"`
	}
	if err = json.Unmarshal(raw, &v); err != nil || len(v.Probabilities) == 0 {
		t.Fatal("missing reference probabilities", err)
	}
	return v.Probabilities
}

// The Go implementation must match onnxruntime running the original model.
func TestModelMatchesOnnxRuntime(t *testing.T) {
	audio, want := readPCM(t, "speech-16k.pcm"), golden(t)
	d, err := NewDetector()
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for i, p := range want {
		got, err := d.Process(audio[i*ChunkSize : (i+1)*ChunkSize])
		if err != nil {
			t.Fatal(err)
		}
		worst = math.Max(worst, math.Abs(float64(got)-p))
	}
	if worst > 1e-4 {
		t.Fatal("deviation from onnxruntime", worst)
	}
	// Reset restores the network to its initial state.
	d.Reset()
	if got, _ := d.Process(audio[:ChunkSize]); math.Abs(float64(got)-want[0]) > 1e-4 {
		t.Fatal("reset did not clear state", got, want[0])
	}
}

func TestDetectorRejectsWrongChunk(t *testing.T) {
	d, _ := NewDetector()
	if _, err := d.Process(make([]float32, 480)); !errors.Is(err, ErrInvalidChunk) {
		t.Fatal(err)
	}
}

// Neither silence nor a pure tone is speech.
func TestModelRejectsSilenceAndTone(t *testing.T) {
	d, _ := NewDetector()
	chunk := make([]float32, ChunkSize)
	for i := 0; i < 30; i++ {
		if p, _ := d.Process(chunk); p > 0.2 {
			t.Fatal("silence classified as speech", p)
		}
	}
	d.Reset()
	for i := 0; i < 30; i++ {
		for j := range chunk {
			chunk[j] = 0.3 * float32(math.Sin(2*math.Pi*440*float64(i*ChunkSize+j)/SampleRate))
		}
		if p, _ := d.Process(chunk); p > 0.5 {
			t.Fatal("tone classified as speech", p)
		}
	}
}

// Weights are shared and read-only: concurrent detectors do not affect each other.
func TestDetectorsAreIndependent(t *testing.T) {
	audio, want := readPCM(t, "speech-16k.pcm"), golden(t)
	done := make(chan float64, 4)
	for range 4 {
		go func() {
			d, _ := NewDetector()
			worst := 0.0
			for i, p := range want {
				got, _ := d.Process(audio[i*ChunkSize : (i+1)*ChunkSize])
				worst = math.Max(worst, math.Abs(float64(got)-p))
			}
			done <- worst
		}()
	}
	for range 4 {
		if worst := <-done; worst > 1e-4 {
			t.Fatal("concurrent detector diverged", worst)
		}
	}
}

func BenchmarkDetector(b *testing.B) {
	audio := readPCM(b, "speech-16k.pcm")
	d, _ := NewDetector()
	n := len(audio) / ChunkSize
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		_, _ = d.Process(audio[(i%n)*ChunkSize : (i%n+1)*ChunkSize])
	}
}
