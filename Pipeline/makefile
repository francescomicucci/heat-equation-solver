.PHONY: all
all: pipeline

pipeline: main.cpp parallelHeat.cpp parallelHeat.hpp
	mpic++ main.cpp parallelHeat.cpp parallelHeat.hpp -o main -Wall -O3 -g
	mpiexec -n 1 ./main
	mpiexec -n 2 ./main
	mpiexec -n 4 ./main

.PHONY: clear
clear:
	rm -f main