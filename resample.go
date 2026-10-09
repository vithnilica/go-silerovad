package silerovad

import (
	"errors"
	"math"
)

// ErrSampleRate is returned by NewStream for an unsupported sample rate.
var ErrSampleRate = errors.New("silerovad: sample rate must be between 8000 and 192000 Hz")

// resampler converts a stream to 16 kHz using a polyphase FIR filter: a sinc with
// a Blackman window. The sample rate ratio is reduced to L/M; output sample n lies
// at input position n·M/L with phase (n·M) mod L. The cutoff is 93.75% of the lower
// Nyquist frequency (7.5 kHz when downsampling to 16 kHz) to prevent aliasing.
type resampler struct {
	up, down int64       // L and M
	half     int         // half the filter length in input samples
	taps     [][]float32 // L phases with 2·half+1 coefficients each
	input    []float32   // input starting at absolute index base
	base     int64
	next     int64 // index of the next output sample
}

func gcd(a, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func newResampler(rate int) (*resampler, error) {
	if rate < 8000 || rate > 192000 {
		return nil, ErrSampleRate
	}
	g := gcd(int64(rate), SampleRate)
	r := &resampler{up: SampleRate / g, down: int64(rate) / g}
	// Downsampling uses a longer filter to keep the transition band equally narrow.
	// Upsampling (8–16 kHz) needs a longer filter because the cutoff is close to the speech band's 3.4 kHz upper edge.
	r.half = 48
	if rate > SampleRate {
		r.half = int(math.Ceil(16 * float64(rate) / SampleRate))
	}
	cutoff := 0.5 * math.Min(float64(rate), SampleRate) * 0.9375 / float64(rate)
	r.taps = make([][]float32, r.up)
	for phase := range r.taps {
		offset := float64(phase) / float64(r.up)
		values := make([]float64, 2*r.half+1)
		sum := 0.0
		for k := -r.half; k <= r.half; k++ {
			t := float64(k) - offset
			if math.Abs(t) >= float64(r.half) {
				continue
			}
			sinc := 2 * cutoff
			if t != 0 {
				sinc = math.Sin(2*math.Pi*cutoff*t) / (math.Pi * t)
			}
			x := (t/float64(r.half) + 1) / 2
			window := 0.42 - 0.5*math.Cos(2*math.Pi*x) + 0.08*math.Cos(4*math.Pi*x)
			values[k+r.half] = sinc * window
			sum += values[k+r.half]
		}
		// Each phase has unity DC gain.
		r.taps[phase] = make([]float32, len(values))
		for i, v := range values {
			r.taps[phase][i] = float32(v / sum)
		}
	}
	// Assume silence before the stream starts; the first output aligns with the first input sample.
	r.input = make([]float32, r.half)
	r.base = -int64(r.half)
	return r, nil
}

// push adds input samples and appends all output samples that can now be computed to out.
func (r *resampler) push(samples []float32, out []float32) []float32 {
	r.input = append(r.input, samples...)
	end := r.base + int64(len(r.input))
	for {
		pos := r.next * r.down
		center := pos / r.up
		if center+int64(r.half) >= end {
			break
		}
		taps := r.taps[pos%r.up]
		window := r.input[center-int64(r.half)-r.base : center+int64(r.half)+1-r.base]
		var sum float32
		for k, v := range taps {
			sum += v * window[k]
		}
		out = append(out, sum)
		r.next++
	}
	// Discard input that is no longer needed for future output.
	if keep := (r.next*r.down)/r.up - int64(r.half); keep > r.base {
		r.input = append(r.input[:0], r.input[keep-r.base:]...)
		r.base = keep
	}
	return out
}

// flush appends the output for the rest of the input, assuming silence after the stream ends.
// The output covers the input duration: ceil(n·L/M) samples for n input samples.
func (r *resampler) flush(out []float32) []float32 {
	total := r.base + int64(len(r.input))
	target := (total*r.up + r.down - 1) / r.down
	out = r.push(make([]float32, r.half+1), out)
	if extra := r.next - target; extra > 0 {
		out = out[:len(out)-int(extra)]
		r.next = target
	}
	return out
}
