# src
# ------------------------------------------------------------------------------------------------

SRC_CPU_SOURCES := $(wildcard $(SRC_DIR)/*.c)
SRC_CPU_OBJECTS := $(patsubst %, $(BUILD_SRC_DIR)/%, $(notdir $(SRC_CPU_SOURCES:.c=.o)))
SRC_CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
SRC_CUDA_OBJECTS := $(patsubst %, $(BUILD_SRC_DIR)/%, $(notdir $(SRC_CUDA_SOURCES:.cu=.o)))

SRC_OBJECTS += $(SRC_CPU_OBJECTS) 
SRC_OBJECTS += $(SRC_CUDA_OBJECTS)

# ------------------------------------------------------------------------------------------------

build_srcs: $(SRC_CPU_OBJECTS) $(SRC_CUDA_OBJECTS)

$(BUILD_SRC_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) -c $< -o $@ $(CC_CFLAGS)

$(BUILD_SRC_DIR)/%.o : $(SRC_DIR)/%.cu
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) -c $< -o $@ $(CC_CFLAGS) $(CC_LDFLAGS)
