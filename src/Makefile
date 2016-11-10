CXX = clang++
TOP = ..
TARGET = $(TOP)/bin/railgun
MAIN = $(TOP)/$(APP)
CXXFLAGS = -Werror
LIBS = -lcudart -lpthread

.PHONY: all clean

all: $(TARGET)

$(TARGET):
	mkdir -p "$$(dirname $(TARGET))"
	$(CXX) $(CXXFLAGS) --cuda-path=/opt/cuda -L/opt/cuda/lib64 -I./ -o $(TARGET) *.cu $(MAIN) $(LIBS)

clean:
	$(RM) $(TARGET)
