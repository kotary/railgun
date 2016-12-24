CXX = clang++
TOP = ..
TARGET = $(TOP)/bin/railgun
MAIN = $(TOP)/$(APP)
OBJGEN = $(TOP)/scripts/defunc.rb
OBJDIR = obj
# EXSRC = $(OBJDIR)/defunc.cu
# EXOBJ = $(EXSRC:.cu=.o)
CXXFLAGS = -Werror --cuda-path=/opt/cuda
LIBS = -lcudart -lpthread -lgc


all: $(TARGET)

.PHONY: all

#$(TARGET): $(EXOBJ)
$(TARGET):
	mkdir -p "$$(dirname $(TARGET))"
	# $(CXX) $(CXXFLAGS) -L/opt/cuda/lib64 -I./ -o $(TARGET) *.cu $(MAIN) $(EXOBJ) $(LIBS)
	$(CXX) $(CXXFLAGS) -L/opt/cuda/lib64 -I./ -o $(TARGET) *.cu $(MAIN) $(LIBS)

$(EXOBJ):
	mkdir -p $(OBJDIR)
	$(OBJGEN) 4 > $(EXSRC)
	$(CXX) $(CXXFLAGS) -I./ -c $(EXSRC)
	mv $$(basename $(EXOBJ)) $(EXOBJ)
	rm -f defunc*

clean:
	rm -f $(TARGET)
.PHONY: clean
