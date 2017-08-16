# tests
# ----------------------------------------------------------------------------
# compiler

TESTS_CFLAGS = $(CC_CFLAGS) \
               -I$(BUILD_UNITY_DIR)

TESTS_LDFLAGS = $(CC_LDFLAGS) \
                -L$(BUILD_UNITY_DIR) -lunity

# ----------------------------------------------------------------------------
# commands

TESTS_CPU_SOURCES := $(wildcard $(TESTS_DIR)/*.c)
TESTS_CPU_OBJECTS := $(patsubst %, $(BUILD_TESTS_DIR)/%, $(notdir $(TESTS_CPU_SOURCES:.c=.o)))
TESTS_CPU_TARGETS := $(patsubst %, $(BUILD_TESTS_DIR)/%, $(notdir $(TESTS_CPU_OBJECTS:.o=)))

TESTS_CUDA_SOURCES := $(wildcard $(TESTS_DIR)/*.cu)
TESTS_CUDA_OBJECTS := $(patsubst %, $(BUILD_TESTS_DIR)/%, $(notdir $(TESTS_CUDA_SOURCES:.cu=.o)))
TESTS_CUDA_TARGETS := $(patsubst %, $(BUILD_TESTS_DIR)/%, $(notdir $(TESTS_CUDA_OBJECTS:.o=)))

TESTS_TARGETS = $(TESTS_CPU_TARGETS) $(TESTS_CUDA_TARGETS)

# ----------------------------------------------------------------------------

run_tests:
	@echo "$(RED) run tests:$(NC)"
	$(foreach test, $(TESTS_TARGETS), \
          $(test) | grep "FAIL"; \
          echo "$(GREEN) TEST $(test)$(NC)";)	
	
build_tests: $(TESTS_TARGETS)

$(BUILD_TESTS_DIR)/% : $(BUILD_TESTS_DIR)/%.o
	@echo "$(RED)Linking $@ $(NC)"
	$(CC) -o $@ $^ $(SRC_OBJECTS) $(TESTS_CFLAGS) $(TESTS_LDFLAGS)

$(BUILD_TESTS_DIR)/%.o : $(TESTS_DIR)/%.c
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) -c $< -o $@ $(TESTS_CFLAGS)

$(BUILD_TESTS_DIR)/%.o : $(TESTS_DIR)/%.cu
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) -c $< -o $@ $(TESTS_CFLAGS)
