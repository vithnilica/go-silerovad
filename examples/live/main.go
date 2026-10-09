// Program live reads raw mono PCM16 from standard input and reports speech in real time.
//
//	arecord -q -f S16_LE -r 16000 -c 1 | go run ./examples/live
//	ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 48000 -f s16le - | go run ./examples/live -rate 48000
//	sox -q -d -t raw -e signed -b 16 -c 1 -r 44100 - | go run ./examples/live -rate 44100
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	silerovad "github.com/vithnilica/go-silerovad"
)

func main() {
	rate := flag.Int("rate", 16000, "input sample rate in Hz")
	meter := flag.Bool("meter", false, "display the probability for each chunk")
	flag.Parse()
	stream, err := silerovad.NewStream(*rate)
	if err != nil {
		log.Fatal(err)
	}
	segmenter := silerovad.NewSegmenter(silerovad.SegmenterOptions{})
	in := bufio.NewReader(os.Stdin)
	// 20 ms of input; Stream assembles its own 32 ms chunks.
	buf := make([]byte, *rate/50*2)
	for {
		if _, err := io.ReadFull(in, buf); err != nil {
			return
		}
		for _, p := range stream.PushPCM16(buf) {
			if *meter {
				fmt.Printf("%5.2f %s\n", p, strings.Repeat("█", int(p*40)))
			}
			if event, ok := segmenter.Next(p); ok {
				fmt.Println(event)
			}
		}
	}
}
