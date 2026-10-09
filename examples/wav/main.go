// Program wav prints speech segments in a WAV file (PCM16, mono or stereo, any supported sample rate).
//
//	go run ./examples/wav recording.wav
//	go run ./examples/wav -threshold 0.6 -min-silence 300ms recording.wav
package main

import (
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	silerovad "github.com/vithnilica/go-silerovad"
)

func main() {
	threshold := flag.Float64("threshold", 0.5, "speech probability threshold")
	minSilence := flag.Duration("min-silence", 100*time.Millisecond, "silence duration required to end speech")
	minSpeech := flag.Duration("min-speech", 250*time.Millisecond, "discard segments shorter than this duration")
	flag.Parse()
	if flag.NArg() != 1 {
		log.Fatal("usage: wav [options] file.wav")
	}
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	samples, rate, err := readWAV(f)
	if err != nil {
		log.Fatal(err)
	}
	segments, err := silerovad.Segments(samples, rate, silerovad.SegmenterOptions{
		Threshold: float32(*threshold), MinSilence: *minSilence, MinSpeech: *minSpeech,
	})
	if err != nil {
		log.Fatal(err)
	}
	var speech time.Duration
	for _, s := range segments {
		fmt.Printf("%8.3f s – %8.3f s\n", s.Start.Seconds(), s.End.Seconds())
		speech += s.End - s.Start
	}
	total := time.Duration(len(samples)) * time.Second / time.Duration(rate)
	fmt.Printf("speech %.1f s of %.1f s (%d segments)\n", speech.Seconds(), total.Seconds(), len(segments))
}

// readWAV reads a PCM16 WAV file and converts multichannel audio to mono by averaging the channels.
func readWAV(r io.Reader) ([]float32, int, error) {
	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil || string(header[:4]) != "RIFF" || string(header[8:]) != "WAVE" {
		return nil, 0, errors.New("not a WAV file")
	}
	var channels, bits uint16
	var rate uint32
	for {
		var chunk [8]byte
		if _, err := io.ReadFull(r, chunk[:]); err != nil {
			return nil, 0, errors.New("WAV file contains no data")
		}
		size := int64(binary.LittleEndian.Uint32(chunk[4:]))
		switch string(chunk[:4]) {
		case "fmt ":
			if size < 16 || size > 1024 {
				return nil, 0, errors.New("invalid fmt chunk")
			}
			body := make([]byte, size+size%2)
			if _, err := io.ReadFull(r, body); err != nil {
				return nil, 0, err
			}
			if binary.LittleEndian.Uint16(body) != 1 {
				return nil, 0, errors.New("only uncompressed PCM is supported")
			}
			channels, rate, bits = binary.LittleEndian.Uint16(body[2:]), binary.LittleEndian.Uint32(body[4:]), binary.LittleEndian.Uint16(body[14:])
		case "data":
			if bits != 16 || channels == 0 {
				return nil, 0, errors.New("only 16-bit PCM is supported")
			}
			// Streaming writers (e.g. ffmpeg to a pipe) leave the size at 0xFFFFFFFF or set it
			// too large, so the data is read up to the declared size or the end of the file.
			body, err := io.ReadAll(io.LimitReader(r, size))
			if err != nil {
				return nil, 0, err
			}
			frame := int(channels) * 2
			samples := make([]float32, len(body)/frame)
			for i := range samples {
				sum := 0
				for c := 0; c < int(channels); c++ {
					sum += int(int16(binary.LittleEndian.Uint16(body[i*frame+c*2:])))
				}
				samples[i] = float32(sum) / float32(channels) / 32768
			}
			return samples, int(rate), nil
		default:
			if _, err := io.CopyN(io.Discard, r, size+size%2); err != nil {
				return nil, 0, err
			}
		}
	}
}
