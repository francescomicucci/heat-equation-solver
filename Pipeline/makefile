.PHONY: all
all: pipeline

pipeline: pipeline.cpp
	mpic++ pipeline.cpp -o main -Wall -O3 -g
	mpiexec -n 1 ./main
	mpiexec -n 2 ./main
	mpiexec -n 3 ./main
	mpiexec -n 4 ./main

.PHONY: clear
clear:
	rm -f main