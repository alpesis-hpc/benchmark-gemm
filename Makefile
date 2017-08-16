# GEMM
# ----------------------------------------------------------------------------
# configurations

MODE = gpu

# ----------------------------------------------------------------------------
# directories

# project dirs
PROJECT_DIR = gemm
LIB_DIR = $(PROJECT_DIR)/lib
INC_DIR = $(PROJECT_DIR)/inc
SRC_DIR = $(PROJECT_DIR)/src
TESTS_DIR = $(PROJECT_DIR)/tests

# build dirs
BUILD_DIR = _build
BUILD_LIB_DIR = $(BUILD_DIR)/lib
BUILD_INC_DIR = $(BUILD_DIR)/inc
BUILD_SRC_DIR = $(BUILD_DIR)/src
BUILD_TESTS_DIR = $(BUILD_DIR)/tests

# ----------------------------------------------------------------------------
# include

include makefiles/lib.make
include makefiles/inc.make
include makefiles/src.make
include makefiles/tests.make

# ----------------------------------------------------------------------------
# compiler

ifeq ($(MODE), cpu)
  CC = gcc
else ifeq ($(MODE), gpu)
  CC = nvcc
endif

CUDA_INC_DIR = /usr/loca/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64

CC_CFLAGS = -I$(INC_DIR) -I$(CUDA_INC_DIR)
CC_LDFLAGS = -L$(CUDA_LIB_DIR) -lcudart 

# ----------------------------------------------------------------------------
# console

RED = \033[1;31m
GREEN = \033[1;32m
BLUE = \033[1;34m
YELLOW = \033[1;33m
NC = \033[1;0m

# ----------------------------------------------------------------------------
# commands

init:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_LIB_DIR)
	mkdir -p $(BUILD_UNITY_DIR)
	mkdir -p $(BUILD_INC_DIR)
	mkdir -p $(BUILD_SRC_DIR)
	mkdir -p $(BUILD_TESTS_DIR)

help:
	@echo "$(RED)Command List:$(NC)"
	@echo "- $(GREEN)all$(NC): run all steps"
	@echo "- $(GREEN)clean$(NC): clean up the build dir"
	@echo "- $(GREEN)dummy$(NC): for testing make commands"


all: clean init inc lib build run

build: build_srcs build_tests

run: run_tests

clean:
	rm -rf $(BUILD_DIR)

dummy: 
	@echo $(SRC_OBJECTS)
	
