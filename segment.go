package silerovad

import (
	"fmt"
	"time"
)

// ChunkDuration is the duration of one chunk (512 samples at 16 kHz).
const ChunkDuration = 32 * time.Millisecond

// SegmenterOptions configures speech segmentation. Zero values select the defaults
// from the upstream Silero VAD VADIterator.
type SegmenterOptions struct {
	// Threshold is the probability threshold for starting speech; the default is 0.5.
	Threshold float32
	// NegThreshold is the threshold for ending speech; the default is Threshold − 0.15.
	NegThreshold float32
	// MinSilence is the silence duration required to end speech; the default is 100 ms.
	MinSilence time.Duration
	// SpeechPad extends each segment at both ends; the default is 30 ms.
	SpeechPad time.Duration
	// MinSpeech discards segments shorter than this duration before padding; it applies
	// only to Segments and defaults to 250 ms.
	MinSpeech time.Duration
}

func (o SegmenterOptions) withDefaults() SegmenterOptions {
	if o.Threshold == 0 {
		o.Threshold = 0.5
	}
	if o.NegThreshold == 0 {
		o.NegThreshold = o.Threshold - 0.15
	}
	if o.MinSilence == 0 {
		o.MinSilence = 100 * time.Millisecond
	}
	if o.SpeechPad == 0 {
		o.SpeechPad = 30 * time.Millisecond
	}
	if o.MinSpeech == 0 {
		o.MinSpeech = 250 * time.Millisecond
	}
	return o
}

// Event marks the start or end of speech; At is the time since the start of the stream.
type Event struct {
	Start bool
	At    time.Duration
}

func (e Event) String() string {
	if e.Start {
		return fmt.Sprintf("speech start %v", e.At)
	}
	return fmt.Sprintf("speech end %v", e.At)
}

// Segmenter converts per-chunk probabilities into speech start and end events.
// It follows the VADIterator class from Silero VAD: speech starts when the probability
// reaches Threshold and ends after it stays below NegThreshold for MinSilence.
type Segmenter struct {
	opts      SegmenterOptions
	now       time.Duration // end of the last chunk
	triggered bool
	tempEnd   time.Duration
}

// NewSegmenter creates a segmenter with the given options (zero values use defaults).
func NewSegmenter(opts SegmenterOptions) *Segmenter {
	return &Segmenter{opts: opts.withDefaults()}
}

// Next processes the next chunk's probability and returns an event if one occurs.
func (s *Segmenter) Next(probability float32) (Event, bool) {
	s.now += ChunkDuration
	if probability >= s.opts.Threshold && s.tempEnd != 0 {
		s.tempEnd = 0
	}
	if probability >= s.opts.Threshold && !s.triggered {
		s.triggered = true
		return Event{Start: true, At: max(0, s.now-s.opts.SpeechPad-ChunkDuration)}, true
	}
	if probability < s.opts.NegThreshold && s.triggered {
		if s.tempEnd == 0 {
			s.tempEnd = s.now
		}
		if s.now-s.tempEnd < s.opts.MinSilence {
			return Event{}, false
		}
		end := s.tempEnd + s.opts.SpeechPad - ChunkDuration
		s.tempEnd, s.triggered = 0, false
		return Event{At: end}, true
	}
	return Event{}, false
}

// Speaking reports whether the segmenter currently considers the audio to be speech.
func (s *Segmenter) Speaking() bool { return s.triggered }

// Segment is a speech segment in a recording.
type Segment struct {
	Start, End time.Duration
}

// Segments finds speech segments in a complete mono recording at the given sample rate.
// It is a simplified equivalent of get_speech_timestamps: it discards segments shorter
// than MinSpeech (measured before padding), closes any ongoing speech segment at the end
// of the recording, and pads segments by SpeechPad within the recording. When the gap
// between two segments is shorter than twice SpeechPad, it is split between them.
func Segments(samples []float32, rate int, opts SegmenterOptions) ([]Segment, error) {
	stream, err := NewStream(rate)
	if err != nil {
		return nil, err
	}
	opts = opts.withDefaults()
	// Segments are padded at the end, so the segmenter reports unpadded times.
	unpadded := opts
	unpadded.SpeechPad = 0
	segmenter := &Segmenter{opts: unpadded}
	total := time.Duration(len(samples)) * time.Second / time.Duration(rate)
	var out []Segment
	var start time.Duration
	add := func(end time.Duration) {
		if end-start >= opts.MinSpeech {
			out = append(out, Segment{start, end})
		}
	}
	for _, p := range append(stream.PushFloat32(samples), stream.Flush()...) {
		if event, ok := segmenter.Next(p); ok {
			if event.Start {
				start = event.At
			} else {
				add(event.At)
			}
		}
	}
	if segmenter.Speaking() {
		add(total)
	}
	pad := opts.SpeechPad
	for i := range out {
		if i == 0 {
			out[i].Start = max(0, out[i].Start-pad)
		}
		if i == len(out)-1 {
			out[i].End = min(total, out[i].End+pad)
			break
		}
		if gap := out[i+1].Start - out[i].End; gap < 2*pad {
			out[i].End += gap / 2
			out[i+1].Start -= gap / 2
		} else {
			out[i].End += pad
			out[i+1].Start -= pad
		}
	}
	return out, nil
}
