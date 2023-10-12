#!/usr/bin/env bash
gcc -g -Og -std=c99   -pedantic -fopenmp -Wall -Wno-unused-result -o rotseq_og rotseq_og.c
gcc -g -Og -std=c99   -pedantic -fopenmp -Wall -Wno-unused-result -o rotseq_xc rotseq_xc.c
gcc -g -Og -std=c99   -pedantic -fopenmp -Wall -Wno-unused-result -o rotseq_ak rotseq_ak.c
g++ -g -Og -std=c++11 -pedantic -fopenmp -Wall -Wno-unused-result -o rotseq rotseq.cpp
g++ -g -Og -std=c++11 -pedantic -fopenmp -Wall -Wno-unused-result -o rotpar rotpar.cpp
