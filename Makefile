# Compiler setup
CC      := gcc
CCs     := 86
NVCC    := nvcc -arch=sm_$(CCs)
CFLAGS  := -Iinclude -O2
NVFLAGS := -Iinclude -O3 -lineinfo
LDFLAGS := -lcuda -lcudart

# Project structure
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
INC_DIR := include

# Files
C_SRCS  := $(wildcard $(SRC_DIR)/*.c)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS    := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(C_SRCS)) \
           $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))

TARGET  := $(BIN_DIR)/color_segmentation

# stb headers
STB_IMAGE_URL       := https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
STB_IMAGE_WRITE_URL := https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
STB_HEADERS         := $(INC_DIR)/stb_image.h $(INC_DIR)/stb_image_write.h

# Default rule
all: $(STB_HEADERS) $(TARGET)

# Build target
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

# Compile C source
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Create directories if not existing
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Download stb headers if missing
$(INC_DIR)/stb_image.h:
	curl -L -o $@ $(STB_IMAGE_URL)

$(INC_DIR)/stb_image_write.h:
	curl -L -o $@ $(STB_IMAGE_WRITE_URL)

# Clean rule
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
