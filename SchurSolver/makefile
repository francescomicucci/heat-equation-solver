.PHONY: all
all: main1D

main1D: main1D.cpp heat1DSchur.cpp heat1DSchur.hpp
	g++ main1D.cpp heat1DSchur.cpp heat1DSchur.hpp -o main1D -Wall -O3 -g
	./main1D

.PHONY: clear
clear:
	rm -f main1D