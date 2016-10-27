TARGET = bin/railgun

.PHONY: all clean

all clean:
	$(MAKE) -C src $@
