#ifndef COLOR_SEGMENTATION_H
#define COLOR_SEGMENTATION_H

#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>

// Constants
// old ones
#define THREADS_PER_BLOCK 128
#define MAX_PATH_LENGTH 1024
#define SHARED_MEMORY_THRESHOLD 1024
#define MAX_BLOCKS_SHARED 32

#define COLOR_NUMBER 4
#define S_ERROR "[ERROR]"
#define S_WARNING "[WARNING]"
#define MAX_ITER 200
#define EPSILON 1e-4

// Macros
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
    }                                                                          \
  }

#define CHECK_WITH_CODE(call, code)                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      return code;                                                             \
    }                                                                          \
  }

#define CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, error_code)       \
  do {                                                                         \
    cudaError_t err = cudaMalloc(&ptr, size);                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));             \
      cleanup_code;                                                            \
      return error_code;                                                       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_MALLOC(ptr, size, cleanup_code)                             \
  CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, 0)

#define CUDA_CHECK_MEMCPY_WITH_CODE(dst, src, size, kind, cleanup_code,        \
                                    error_code)                                \
  do {                                                                         \
    cudaError_t err = cudaMemcpy(dst, src, size, kind);                        \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));             \
      cleanup_code;                                                            \
      return error_code;                                                       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_MEMCPY(dst, src, size, kind, cleanup_code)                  \
  CUDA_CHECK_MEMCPY_WITH_CODE(dst, src, size, kind, cleanup_code, 0)

// Data structs
typedef struct {
  int width;
  int height;
  int channels;
  char *filename;
  unsigned char *data;
} LoadedImage;

// Functions
void print_help(const char *program_name);
bool is_valid_image(const char *filename);
unsigned char *load_image(char *dir, LoadedImage *img);
void cleanup(float *centroids, int *labels, unsigned char *img_data);
void dump_image(const float *centroids, const int *labels, LoadedImage *img);
char *name_changer(const char *filepath);

#endif // COLOR_SEGMENTATION_H
