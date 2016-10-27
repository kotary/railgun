CXX = clang++
TOP = ..
TARGET = $(TOP)/bin/railgun
CXXFLAGS = -Werror
LIBS = -lcudart -lpthread
SRCS = $(wildcard *.cu)
BCS = $(SRCS:.cu=.bc)
DEPS:=$(SRCS:.cu=.d)

.PHONY: all clean

all: $(TARGET)

%.bc: %.cu
	$(CXX) $(CXXFLAGS) -emit-llvm -MMD -MP -c -O0 --cuda-path=/opt/cuda -o $@ $<

program.bc: $(BCS)
	llvm-link $(BCS) -o program.bc

program.o: program.bc
	llc -filetype=obj program.bc -o program.o

$(TARGET): program.o
	mkdir -p "$$(dirname $(TARGET))"
	$(CXX) $(CXXFLAGS) -L/opt/cuda/lib64 -o $(TARGET) program.o $(LIBS)

clean:
	$(RM) program.o program.bc
	$(RM) $(BCS) $(DEPS)
	$(RM) $(TARGET)

-include $(DEPS)
