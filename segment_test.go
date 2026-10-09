package silerovad

import (
	"testing"
	"time"
)

func events(s *Segmenter, probs ...float32) []Event {
	var out []Event
	for _, p := range probs {
		if e, ok := s.Next(p); ok {
			out = append(out, e)
		}
	}
	return out
}

// Behavior matches VADIterator: a padded start, with brief probability dips not ending speech.
func TestSegmenterFollowsVADIterator(t *testing.T) {
	s := NewSegmenter(SegmenterOptions{})
	got := events(s, 0.1, 0.1, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1)
	if len(got) != 2 || !got[0].Start || got[1].Start {
		t.Fatal("events", got)
	}
	// Speech started in the third chunk (64 ms), minus 30 ms of padding.
	if got[0].At != 64*time.Millisecond-30*time.Millisecond {
		t.Fatal("start", got[0].At)
	}
	// Two chunks below the threshold (64 ms) are shorter than MinSilence (100 ms), so speech continues.
	// End: the first silent chunk ends at 288 ms; add 30 ms of padding and subtract the chunk duration.
	if got[1].At != 288*time.Millisecond+30*time.Millisecond-ChunkDuration {
		t.Fatal("end", got[1].At)
	}
	if s.Speaking() {
		t.Fatal("segmenter still reports speech")
	}
}

// Probabilities between NegThreshold and Threshold neither start nor end speech.
func TestSegmenterHysteresis(t *testing.T) {
	s := NewSegmenter(SegmenterOptions{})
	if got := events(s, 0.4, 0.45, 0.4); len(got) != 0 {
		t.Fatal("speech started below the threshold", got)
	}
	events(s, 0.6)
	if got := events(s, 0.4, 0.4, 0.4, 0.4, 0.4); len(got) != 0 || !s.Speaking() {
		t.Fatal("speech ended above NegThreshold", got)
	}
}

func TestSegmentsOnSpeech(t *testing.T) {
	audio := readPCM(t, "speech-16k.pcm")
	segments, err := Segments(audio, SampleRate, SegmenterOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) == 0 {
		t.Fatal("no speech found in the recording")
	}
	total := time.Duration(len(audio)) * time.Second / SampleRate
	for i, s := range segments {
		if s.Start < 0 || s.End > total || s.End-s.Start < 250*time.Millisecond {
			t.Fatal("invalid segment", s)
		}
		if i > 0 && s.Start < segments[i-1].End {
			t.Fatal("segments overlap", segments)
		}
	}
}
