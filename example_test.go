package silerovad_test

import (
	"encoding/binary"
	"fmt"
	"os"
	"time"

	silerovad "github.com/vithnilica/go-silerovad"
)

func loadPCM16(name string) []byte {
	raw, err := os.ReadFile(name)
	if err != nil {
		panic(err)
	}
	return raw
}

func ExampleDetector() {
	d, err := silerovad.NewDetector()
	if err != nil {
		panic(err)
	}
	// A chunk of 512 samples at 16 kHz in the range −1…1; silence in this example.
	chunk := make([]float32, silerovad.ChunkSize)
	p, err := d.Process(chunk)
	if err != nil {
		panic(err)
	}
	fmt.Println(p < 0.5)
	// Output: true
}

func ExampleStream() {
	// Microphone audio at 24 kHz, for example from an Opus decoder in 20 ms frames.
	pcm := loadPCM16("testdata/speech-24k.pcm")
	stream, err := silerovad.NewStream(24000)
	if err != nil {
		panic(err)
	}
	speech := 0
	for i := 0; i+960 <= len(pcm); i += 960 {
		for _, p := range stream.PushPCM16(pcm[i : i+960]) {
			if p >= 0.5 {
				speech++
			}
		}
	}
	fmt.Printf("speech in %d of %d chunks of 32 ms\n", speech, stream.Chunks())
	// Output: speech in 136 of 186 chunks of 32 ms
}

func ExampleSegmenter() {
	pcm := loadPCM16("testdata/speech-16k.pcm")
	stream, _ := silerovad.NewStream(silerovad.SampleRate)
	segmenter := silerovad.NewSegmenter(silerovad.SegmenterOptions{MinSilence: 300 * time.Millisecond})
	samples := make([]float32, len(pcm)/2)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768
	}
	for _, p := range stream.PushFloat32(samples) {
		if event, ok := segmenter.Next(p); ok {
			fmt.Println(event)
		}
	}
	// Output:
	// speech start 0s
	// speech end 2.846s
	// speech start 3.17s
	// speech end 4.862s
	// speech start 5.314s
}

func ExampleSegments() {
	pcm := loadPCM16("testdata/speech-16k.pcm")
	samples := make([]float32, len(pcm)/2)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768
	}
	segments, err := silerovad.Segments(samples, silerovad.SampleRate, silerovad.SegmenterOptions{})
	if err != nil {
		panic(err)
	}
	for _, s := range segments {
		fmt.Printf("%.3f–%.3f s\n", s.Start.Seconds(), s.End.Seconds())
	}
	// Output:
	// 0.000–0.350 s
	// 0.450–2.078 s
	// 2.274–2.846 s
	// 3.170–4.862 s
	// 5.314–5.984 s
}
