#!/usr/bin/env bash
seq='gcc -g -pedantic -std=c99 -Wall -o dist_seq dist_seq.c'
par='nvcc dist_par.cu -o dist_par'

echo "$seq"
eval "$seq"

echo "$par"
eval "$par"
