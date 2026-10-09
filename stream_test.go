package silerovad

import (
	"math"
	"os"
	"testing"
)

// Input at 16 kHz reaches the network unchanged, even when chunk boundaries do not align.
func TestStream16kMatchesDetector(t *testing.T) {
	audio, want := readPCM(t, "speech-16k.pcm"), golden(t)
	s, err := NewStream(SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	var got []float32
	for i := 0; i < len(audio); i += 300 {
		got = append(got, s.PushFloat32(audio[i:min(i+300, len(audio))])...)
	}
	if len(got) != len(want) || s.Chunks() != int64(len(want)) {
		t.Fatal("chunk count", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i])-want[i]) > 1e-4 {
			t.Fatal("chunk", i, got[i], want[i])
		}
	}
}

// The full 24 kHz → 16 kHz → network pipeline matches onnxruntime on the original 16 kHz recording.
func TestStream24kMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/speech-24k.pcm")
	if err != nil {
		t.Fatal(err)
	}
	want := golden(t)
	s, _ := NewStream(24000)
	var got []float32
	// 20 ms frames, as from an Opus decoder: 960 bytes of PCM16.
	for i := 0; i < len(raw); i += 960 {
		got = append(got, s.PushPCM16(raw[i:min(i+960, len(raw))])...)
	}
	if len(got) < len(want)-1 {
		t.Fatal("too few chunks", len(got), len(want))
	}
	agree, diff := 0, 0.0
	n := min(len(got), len(want))
	for i := range got[:n] {
		if (got[i] > 0.5) == (want[i] > 0.5) {
			agree++
		}
		diff += math.Abs(float64(got[i]) - want[i])
	}
	if float64(agree)/float64(n) < 0.97 || diff/float64(n) > 0.03 {
		t.Fatal("resampled pipeline differs", float64(agree)/float64(n), diff/float64(n))
	}
}

func BenchmarkStream24k(b *testing.B) {
	raw, _ := os.ReadFile("testdata/speech-24k.pcm")
	s, _ := NewStream(24000)
	b.SetBytes(960)
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		off := (i * 960) % (len(raw) - 960)
		s.PushPCM16(raw[off : off+960])
	}
}
