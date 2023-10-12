#!/usr/bin/env bash
for i in "$@"; do
	echo "======== Example $i ========"
	./rotseq_og "input$i.txt" /dev/null && gprof --flat-profile ./rotseq_og > "profiling/og$i.txt"
	./rotseq "input$i.txt" /dev/null && gprof --flat-profile ./rotseq > "profiling/s$i.txt"
	./rotpar "input$i.txt" /dev/null && gprof --flat-profile ./rotpar > "profiling/p$i.txt"
done
