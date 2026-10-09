package silerovad

import (
	"errors"
	"math"
	"testing"
)

func sine(freq float64, rate, samples int) []float32 {
	out := make([]float32, samples)
	for i := range out {
		out[i] = 0.5 * float32(math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return out
}

func rms(v []float32) float64 {
	sum := 0.0
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum / float64(len(v)))
}

// Resampling preserves the speech band and attenuates frequencies above 8 kHz to prevent aliasing.
func TestResamplerFrequencyResponse(t *testing.T) {
	for _, rate := range []int{8000, 22050, 24000, 44100, 48000} {
		for _, c := range []struct {
			freq     float64
			min, max float64
		}{{300, 0.97, 1.03}, {1000, 0.97, 1.03}, {3400, 0.97, 1.03}, {10000, 0, 0.01}} {
			if c.freq >= float64(rate)/2 {
				continue
			}
			r, err := newResampler(rate)
			if err != nil {
				t.Fatal(err)
			}
			out := r.push(sine(c.freq, rate, rate), nil)
			// The final samples are waiting for future filter input.
			lag := int(math.Ceil(float64(r.half) * SampleRate / float64(rate)))
			if len(out) < SampleRate-lag-1 || len(out) > SampleRate {
				t.Fatal(rate, "Hz: 1 s of input produced", len(out), "samples")
			}
			gain := rms(out[1000:15000]) / (0.5 / math.Sqrt2)
			if gain < c.min || gain > c.max {
				t.Fatal(rate, "Hz: gain at", c.freq, "Hz is", gain)
			}
		}
	}
}

// The result does not depend on the size of the input chunks.
func TestResamplerChunkingInvariant(t *testing.T) {
	input := sine(700, 44100, 44100)
	whole, _ := newResampler(44100)
	want := whole.push(input, nil)
	pieces, _ := newResampler(44100)
	var got []float32
	for i := 0; i < len(input); i += 37 {
		got = pieces.push(input[i:min(i+37, len(input))], got)
	}
	if len(got) != len(want) {
		t.Fatal("sample count mismatch", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatal("sample mismatch", i)
		}
	}
}

func TestResamplerRejectsRate(t *testing.T) {
	for _, rate := range []int{0, 4000, 400000} {
		if _, err := NewStream(rate); !errors.Is(err, ErrSampleRate) {
			t.Fatal(rate, err)
		}
	}
}
