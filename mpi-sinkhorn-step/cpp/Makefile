CXX = mpicxx
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra -pedantic

BIN := sinkhorn_cpp
SRC := sinkhorn_cpp.cpp

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(BIN)
