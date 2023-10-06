CC := nvcc
CPPFLAGS :=

EIGEN_PATH := deps/eigen-3.4.0 # change this as necessary

SRCFILES = $(shell ls src/*)

INCLUDES := -Iinclude # local path
INCLUDES += -I$(EIGEN_PATH)

all: test timing

test:
	$(CC) $(CPPFLAGS) $(INCLUDES) $(SRCFILES) $@.cpp -o $@

timing:
	$(CC) $(CPPFLAGS) $(INCLUDES) $(SRCFILES) $@.cpp -o $@

clean:
	rm -f test timing

.PHONY: all clean
