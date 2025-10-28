#ifndef GPU_KMEANS_H
#define GPU_KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define THREADS_PER_BLOCK 256
#define MAX_PATH_LENGTH 1024
// Maximum number of floats (n_colors * n_channels) to attempt caching in shared
// memory.
#define SHARED_MEMORY_THRESHOLD 4096
#define MAX_BLOCKS_SHARED 32
#define DEFAULT_STREAMS 4

// Macros
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = (call);                                          \
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
  CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, -1.0)

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
  CUDA_CHECK_MEMCPY_WITH_CODE(dst, src, size, kind, cleanup_code, -1.0)

/**
 * gpu_kmeans_image_colors - Performs k-means clustering on image pixel colors
 * using GPU
 * @data: Contiguous array of bytes representing image data (n_channels *
 * n_pixels)
 * @n_pixels: Total number of pixels in the image
 * @n_channels: Number of color channels per pixel (e.g., 3 for RGB, 4 for RGBA)
 * @n_colors: Number of clusters (k) to find in the color space
 * @centroids: Output array of shape (n_colors * n_channels) for cluster centers
 * @labels: Output array of shape (n_pixels) with cluster assignment for each
 * pixel
 * @max_iter: Maximum number of iterations for convergence
 * @tolerance: Convergence threshold for centroid movement
 *
 * Return: Final inertia value (sum of squared distances), or -1.0 on error
 */
double gpu_kmeans_image_colors(const unsigned char *data, int n_pixels,
                               int n_channels, int n_colors, float *centroids,
                               int *labels, int max_iter, float tolerance);

/**
 * kmeans_pp_init_gpu - Initializes centroids using k-means++ algorithm on GPU
 * @d_pixels: Device array of shape (n_pixels * n_channels) containing pixel
 * data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @n_colors: Number of centroids to initialize
 * @d_centroids: Output device array of shape (n_colors * n_channels) for
 * initial centroids
 * @h_pixels: Host array of pixel data (for random sampling)
 */
void kmeans_pp_init_gpu(const float *d_pixels, int n_pixels, int n_channels,
                        int n_colors, float *d_centroids,
                        const float *h_pixels);

// CUDA kernel declarations (callable from .cu files only)
#ifdef __CUDACC__

/**
 * euclidean_distance_squared - Computes squared Euclidean distance between
 * pixel and centroid
 * @pixel: Pointer to pixel data (n_channels floats)
 * @centroid: Pointer to centroid data (n_channels floats)
 * @n_channels: Number of color channels
 *
 * Return: Squared distance as float
 */
__device__ float euclidean_distance_squared(const float *pixel,
                                            const float *centroid,
                                            int n_channels);

/**
 * assign_clusters_kernel - CUDA kernel to assign each pixel to nearest centroid
 * @pixels: Device array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @centroids: Device array of shape (n_colors * n_channels) containing cluster
 * centers
 * @n_colors: Number of clusters
 * @labels: Output device array of shape (n_pixels) with cluster indices
 */
__global__ void assign_clusters_kernel(const float *pixels, int n_pixels,
                                       int n_channels, const float *centroids,
                                       int n_colors, int *labels);

/**
 * assign_clusters_shared_kernel - Optimized kernel using shared memory for
 * centroids
 * @pixels: Device array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @centroids: Device array of shape (n_colors * n_channels) containing cluster
 * centers
 * @n_colors: Number of clusters
 * @labels: Output device array of shape (n_pixels) with cluster indices
 */
__global__ void assign_clusters_shared_kernel(const float *pixels, int n_pixels,
                                              int n_channels,
                                              const float *centroids,
                                              int n_colors, int *labels);

/**
 * compute_new_centroids_kernel - CUDA kernel to compute sum and count for each
 * cluster
 * @pixels: Device array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @labels: Device array of shape (n_pixels) with cluster assignments
 * @n_colors: Number of clusters
 * @cluster_sums: Output device array of shape (n_colors * n_channels) for sum
 * accumulation
 * @cluster_counts: Output device array of shape (n_colors) for count
 * accumulation
 */
__global__ void compute_new_centroids_kernel(const float *pixels, int n_pixels,
                                             int n_channels, const int *labels,
                                             int n_colors, float *cluster_sums,
                                             int *cluster_counts);

/**
 * finalize_centroids_kernel - CUDA kernel to divide sums by counts to get new
 * centroids
 * @cluster_sums: Device array of shape (n_colors * n_channels) with accumulated
 * sums
 * @cluster_counts: Device array of shape (n_colors) with accumulated counts
 * @n_colors: Number of clusters
 * @n_channels: Number of channels per pixel
 * @old_centroids: Device array of previous centroids (for empty clusters)
 * @new_centroids: Output device array of shape (n_colors * n_channels) for new
 * centroids
 */
__global__ void finalize_centroids_kernel(const float *cluster_sums,
                                          const int *cluster_counts,
                                          int n_colors, int n_channels,
                                          const float *old_centroids,
                                          float *new_centroids);

/**
 * calculate_inertia_kernel - CUDA kernel to compute partial inertia for each
 * pixel
 * @pixels: Device array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @centroids: Device array of shape (n_colors * n_channels) containing cluster
 * centers
 * @labels: Device array of shape (n_pixels) with cluster assignments
 * @partial_inertia: Output device array for partial inertia values (one per
 * block)
 */
__global__ void calculate_inertia_kernel(const float *pixels, int n_pixels,
                                         int n_channels, const float *centroids,
                                         const int *labels,
                                         double *partial_inertia);

#endif // __CUDACC__

#ifdef __cplusplus
}
#endif

#endif // GPU_KMEANS_H
