#!/usr/bin/env bash
seq='gcc -O2 -pedantic -Wall -o remove0_seq remove0_seq.c'
par='mpicc -g0 -O2 -pedantic -Wall -o remove0_par remove0_par.c'

echo "$seq"
eval "$seq"

echo "$par"
eval "$par"
