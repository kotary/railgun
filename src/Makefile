CXX = clang++
TOP = ..
TARGET = $(TOP)/bin/railgun
CXXFLAGS = -Werror
LIBS = -lcudart -lpthread

.PHONY: all clean

all: $(TARGET)

$(TARGET):
	mkdir -p "$$(dirname $(TARGET))"
	$(CXX) $(CXXFLAGS) --cuda-path=/opt/cuda -L/opt/cuda/lib64 -o $(TARGET) *.cu $(LIBS)

clean:
	$(RM) $(TARGET)
