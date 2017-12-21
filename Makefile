PROJECT_DIR = gemm
BUILD_DIR = _build

HEADER_DIR = $(PROJECT_DIR)
SOURCES = $(PROJECT_DIR)/*.c

CC = gcc
CC_FLAGS = -I$(HEADER_DIR)

all: clean build
	$(CC) $(CC_FLAGS) -o $(BUILD_DIR)/test_gemm_cpu $(SOURCES)
	$(BUILD_DIR)/test_gemm_cpu

build:
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
