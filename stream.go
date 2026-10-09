package silerovad

import "encoding/binary"

// Stream processes continuous audio at any supported sample rate.
// It resamples audio to 16 kHz, assembles chunks of 512 samples, and returns a speech
// probability for each complete chunk. Input can be supplied in chunks of any size.
// It is not safe for concurrent use.
type Stream struct {
	detector  *Detector
	resampler *resampler // nil for 16 kHz input
	pending   []float32
	scratch   []float32
	odd       []byte // a trailing byte from PushPCM16 waiting for its pair
	chunks    int64
}

// NewStream creates a stream for mono audio at the given sample rate (8–192 kHz).
func NewStream(rate int) (*Stream, error) {
	d, err := NewDetector()
	if err != nil {
		return nil, err
	}
	s := &Stream{detector: d}
	if rate != SampleRate {
		if s.resampler, err = newResampler(rate); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// PushFloat32 accepts samples in the range −1…1 and returns probabilities for complete chunks.
func (s *Stream) PushFloat32(samples []float32) []float32 {
	if s.resampler == nil {
		s.pending = append(s.pending, samples...)
	} else {
		s.pending = s.resampler.push(samples, s.pending)
	}
	return s.process()
}

// process evaluates all complete chunks and keeps the remainder for the next call.
func (s *Stream) process() []float32 {
	var probabilities []float32
	i := 0
	for ; len(s.pending)-i >= ChunkSize; i += ChunkSize {
		probabilities = append(probabilities, s.detector.process(s.pending[i:i+ChunkSize]))
		s.chunks++
	}
	s.pending = append(s.pending[:0], s.pending[i:]...)
	return probabilities
}

// Flush ends the stream: it processes the audio still held by the resampler and pads
// the last incomplete chunk with silence. It returns the remaining probabilities
// (at most two). Call it once after the last push; the stream cannot be used afterwards.
func (s *Stream) Flush() []float32 {
	s.odd = nil
	if s.resampler != nil {
		s.pending = s.resampler.flush(s.pending)
	}
	probabilities := s.process()
	if len(s.pending) > 0 {
		s.pending = append(s.pending, make([]float32, ChunkSize-len(s.pending))...)
		probabilities = append(probabilities, s.process()...)
	}
	return probabilities
}

// PushPCM16 accepts little-endian mono PCM16, as used by WAV files and most microphones.
// The input may have an odd length; a trailing byte is kept until the next call.
func (s *Stream) PushPCM16(pcm []byte) []float32 {
	s.scratch = s.scratch[:0]
	if len(s.odd) > 0 && len(pcm) > 0 {
		s.scratch = append(s.scratch, float32(int16(binary.LittleEndian.Uint16([]byte{s.odd[0], pcm[0]})))/32768)
		s.odd, pcm = s.odd[:0], pcm[1:]
	}
	for ; len(pcm) >= 2; pcm = pcm[2:] {
		s.scratch = append(s.scratch, float32(int16(binary.LittleEndian.Uint16(pcm)))/32768)
	}
	if len(pcm) == 1 {
		s.odd = append(s.odd[:0], pcm[0])
	}
	return s.PushFloat32(s.scratch)
}

// Chunks returns the number of chunks processed so far; chunk i starts at i × 32 ms.
func (s *Stream) Chunks() int64 { return s.chunks }
