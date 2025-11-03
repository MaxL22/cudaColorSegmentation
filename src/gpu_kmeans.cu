#include "../include/gpu_kmeans.h"
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * euclidean_distance_squared: Computes squared Euclidean distance between
 * pixel and centroid
 * @pixel: Pointer to pixel data (n_channels floats)
 * @centroid: Pointer to centroid data (n_channels floats)
 * @n_channels: Number of color channels
 *
 * Return: Squared distance as float
 */
__device__ float euclidean_distance_squared(const float *pixel,
                                            const float *centroid,
                                            int n_channels) {
  float dist = 0.0f;

#pragma unroll
  for (int c = 0; c < 4; c++) {
    if (c < n_channels) {
      float diff = pixel[c] - centroid[c];
      dist += diff * diff;
    }
  }
  return dist;
}

/**
 * assign_clusters_kernel: kernel to assign each pixel to nearest centroid,
 * without shared memory
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
                                       int n_colors, int *labels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pixels)
    return;

  // Load pixel into registers
  // Not using n_channels to unroll, it's 4 or less, doesn't make much of a
  // difference
  float pixel[4];
#pragma unroll
  for (int c = 0; c < 4; c++) {
    if (c < n_channels) {
      pixel[c] = pixels[idx * n_channels + c];
    }
  }

  // Find nearest centroid
  float min_dist = FLT_MAX;
  int best_cluster = 0;
  for (int k = 0; k < n_colors; k++) {
    float dist = euclidean_distance_squared(pixel, &centroids[k * n_channels],
                                            n_channels);
    if (dist < min_dist) {
      min_dist = dist;
      best_cluster = k;
    }
  }

  labels[idx] = best_cluster;
}

/**
 * assign_clusters_shared_kernel: kernel using shared memory for centroids
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
                                              int n_colors, int *labels) {
  extern __shared__ float shared_centroids[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load centroids into shared memory
  for (int i = threadIdx.x; i < n_colors * n_channels; i += blockDim.x) {
    shared_centroids[i] = centroids[i];
  }
  __syncthreads();

  if (idx >= n_pixels)
    return;

  // Load pixel into registers
  float pixel[4];
#pragma unroll
  for (int c = 0; c < 4; c++) {
    if (c < n_channels) {
      pixel[c] = pixels[idx * n_channels + c];
    }
  }

  // Find nearest centroid
  float min_dist = FLT_MAX;
  int best_cluster = 0;
  for (int k = 0; k < n_colors; k++) {
    float dist = euclidean_distance_squared(
        pixel, &shared_centroids[k * n_channels], n_channels);
    if (dist < min_dist) {
      min_dist = dist;
      best_cluster = k;
    }
  }

  labels[idx] = best_cluster;
}

/**
 * compute_new_centroids_kernel: compute sum and count for each cluster
 * @pixels: Device array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel (max 4)
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
                                             int *cluster_counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load pixel data and cluster assignment
  int cluster = -1;
  float pixel_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (idx < n_pixels) {
    cluster = labels[idx];
    int base = idx * n_channels;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c < n_channels) {
        pixel_data[c] = pixels[base + c];
      }
    }
  }

  // Warp-level aggregation
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  // Process each cluster that appears in this warp
  for (int target_cluster = 0; target_cluster < n_colors; target_cluster++) {
    // Check which threads in warp have this cluster
    unsigned mask = __ballot_sync(0xFFFFFFFF, cluster == target_cluster);

    if (mask == 0)
      continue; // No threads in this warp have this cluster

    // Aggregate within warp using shuffle operations
    float warp_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int warp_count = 0;

    if (cluster == target_cluster) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        warp_sums[c] = pixel_data[c];
      }
      warp_count = 1;
    }

// Reduce across warp, sum up values from all lanes with this cluster
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        warp_sums[c] += __shfl_down_sync(mask, warp_sums[c], offset);
      }
      warp_count += __shfl_down_sync(mask, warp_count, offset);
    }

    // Only lane 0 (or first active lane) does the atomic operation
    int first_lane = __ffs(mask) - 1;
    if (lane_id == first_lane) {
      int base_cluster = target_cluster * n_channels;
#pragma unroll
      for (int c = 0; c < 4; c++) {
        if (c < n_channels && warp_sums[c] != 0.0f) {
          atomicAdd(&cluster_sums[base_cluster + c], warp_sums[c]);
        }
      }
      if (warp_count > 0) {
        atomicAdd(&cluster_counts[target_cluster], warp_count);
      }
    }
  }
}

/**
 * finalize_centroids_kernel: kernel to divide sums by counts to get new
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
                                          float *new_centroids) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_colors)
    return;

  if (cluster_counts[k] > 0) {
// Average to get new centroid
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c < n_channels)
        new_centroids[k * n_channels + c] =
            cluster_sums[k * n_channels + c] / cluster_counts[k];
    }
  } else {
// Keep old centroid for empty clusters
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c < n_channels)
        new_centroids[k * n_channels + c] = old_centroids[k * n_channels + c];
    }
  }
}

/**
 * calculate_inertia_kernel: kernel to compute partial inertia for each
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
                                         double *partial_inertia) {
  __shared__ double shared_inertia[THREADS_PER_BLOCK];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  double local_inertia = 0.0;

  if (idx < n_pixels) {
    int cluster = labels[idx];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c < n_channels) {
        double diff =
            pixels[idx * n_channels + c] - centroids[cluster * n_channels + c];
        local_inertia += diff * diff;
      }
    }
  }

  shared_inertia[tid] = local_inertia;
  __syncthreads();

  // Reduction shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_inertia[tid] += shared_inertia[tid + s];
    }
    __syncthreads();
  }

  // First thread writes block result
  if (tid == 0) {
    partial_inertia[blockIdx.x] = shared_inertia[0];
  }
}

/**
 * kmeans_pp_init_gpu: initializes centroids using k-means++ algorithm
 * @d_pixels: Device array of shape (n_pixels * n_channels) containing pixel
 * data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @n_colors: Number of centroids to initialize
 * @d_centroids: Output device array of shape (n_colors * n_channels) for
 * initial centroids
 * @h_pixels: Host array of pixel data (for random sampling)
 *
 * It's not slow per se, it's probably faster to keep the CPU version
 */
void kmeans_pp_init_gpu(const float *d_pixels, int n_pixels, int n_channels,
                        int n_colors, float *d_centroids,
                        const float *h_pixels) {
  float *h_centroids = (float *)malloc(n_colors * n_channels * sizeof(float));
  float *distances = (float *)malloc(n_pixels * sizeof(float));

  // Choose first centroid randomly
  int first_idx = rand() % n_pixels;
  for (int c = 0; c < n_channels; c++) {
    h_centroids[c] = h_pixels[first_idx * n_channels + c];
  }

  // Choose remaining centroids
  for (int k = 1; k < n_colors; k++) {
    // Calculate minimum distance to existing centroids (on CPU for simplicity)
    for (int i = 0; i < n_pixels; i++) {
      float min_dist = FLT_MAX;
      for (int j = 0; j < k; j++) {
        float dist = 0.0f;
        for (int c = 0; c < n_channels; c++) {
          float diff =
              h_pixels[i * n_channels + c] - h_centroids[j * n_channels + c];
          dist += diff * diff;
        }
        if (dist < min_dist) {
          min_dist = dist;
        }
      }
      distances[i] = min_dist;
    }

    // Choose next centroid with probability proportional to distance squared
    float total_dist = 0.0f;
    for (int i = 0; i < n_pixels; i++) {
      total_dist += distances[i];
    }

    float r = ((float)rand() / RAND_MAX) * total_dist;
    float cumsum = 0.0f;
    int chosen_idx = 0;
    for (int i = 0; i < n_pixels; i++) {
      cumsum += distances[i];
      if (cumsum >= r) {
        chosen_idx = i;
        break;
      }
    }

    for (int c = 0; c < n_channels; c++) {
      h_centroids[k * n_channels + c] = h_pixels[chosen_idx * n_channels + c];
    }
  }

  // Copy to device
  CHECK(cudaMemcpy(d_centroids, h_centroids,
                   n_colors * n_channels * sizeof(float),
                   cudaMemcpyHostToDevice));

  free(h_centroids);
  free(distances);
}

/**
 * gpu_kmeans_image_colors: k-means clustering on image pixel colors using GPU
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
                               int *labels, int max_iter, float tolerance) {
  // Allocate host memory
  float *h_pixels = (float *)malloc(n_pixels * n_channels * sizeof(float));

  // Convert byte data to float
  for (int i = 0; i < n_pixels * n_channels; i++) {
    h_pixels[i] = (float)data[i];
  }

  // Allocate device memory
  float *d_pixels, *d_centroids, *d_new_centroids, *d_cluster_sums;
  int *d_labels, *d_cluster_counts;
  double *d_partial_inertia;

  CUDA_CHECK_MALLOC(d_pixels, n_pixels * n_channels * sizeof(float),
                    free(h_pixels));
  CUDA_CHECK_MALLOC(d_centroids, n_colors * n_channels * sizeof(float), {
    cudaFree(d_pixels);
    free(h_pixels);
  });
  CUDA_CHECK_MALLOC(d_new_centroids, n_colors * n_channels * sizeof(float), {
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    free(h_pixels);
  });
  CUDA_CHECK_MALLOC(d_labels, n_pixels * sizeof(int), {
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    free(h_pixels);
  });
  CUDA_CHECK_MALLOC(d_cluster_sums, n_colors * n_channels * sizeof(float), {
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_labels);
    free(h_pixels);
  });
  CUDA_CHECK_MALLOC(d_cluster_counts, n_colors * sizeof(int), {
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_labels);
    cudaFree(d_cluster_sums);
    free(h_pixels);
  });

  int blocks = (n_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  CUDA_CHECK_MALLOC(d_partial_inertia, blocks * sizeof(double), {
    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_labels);
    cudaFree(d_cluster_sums);
    cudaFree(d_cluster_counts);
    free(h_pixels);
  });

  // Copy pixels to device
  CUDA_CHECK_MEMCPY(d_pixels, h_pixels, n_pixels * n_channels * sizeof(float),
                    cudaMemcpyHostToDevice, {
                      cudaFree(d_pixels);
                      cudaFree(d_centroids);
                      cudaFree(d_new_centroids);
                      cudaFree(d_labels);
                      cudaFree(d_cluster_sums);
                      cudaFree(d_cluster_counts);
                      cudaFree(d_partial_inertia);
                      free(h_pixels);
                    });

  // Initialize centroids using k-means++
  kmeans_pp_init_gpu(d_pixels, n_pixels, n_channels, n_colors, d_centroids,
                     h_pixels);

  // Setup kernel launch parameters
  int centroid_blocks = (n_colors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Determine if we can use shared memory for centroids
  size_t shared_mem_size = n_colors * n_channels * sizeof(float);
  bool use_shared =
      (shared_mem_size <= SHARED_MEMORY_THRESHOLD * sizeof(float));

  // Main k-means loop
  for (int iter = 0; iter < max_iter; iter++) {
    // Assign clusters
    if (use_shared) {
      assign_clusters_shared_kernel<<<blocks, THREADS_PER_BLOCK,
                                      shared_mem_size>>>(
          d_pixels, n_pixels, n_channels, d_centroids, n_colors, d_labels);
    } else {
      assign_clusters_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          d_pixels, n_pixels, n_channels, d_centroids, n_colors, d_labels);
    }
    cudaDeviceSynchronize();

    // Reset cluster sums and counts
    CHECK(cudaMemset(d_cluster_sums, 0, n_colors * n_channels * sizeof(float)));
    CHECK(cudaMemset(d_cluster_counts, 0, n_colors * sizeof(int)));

    // Compute new centroids
    compute_new_centroids_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_pixels, n_pixels, n_channels, d_labels, n_colors, d_cluster_sums,
        d_cluster_counts);
    cudaDeviceSynchronize();

    finalize_centroids_kernel<<<centroid_blocks, THREADS_PER_BLOCK>>>(
        d_cluster_sums, d_cluster_counts, n_colors, n_channels, d_centroids,
        d_new_centroids);
    cudaDeviceSynchronize();

    // Check for convergence (copy centroids to host for comparison)
    float *h_old_centroids =
        (float *)malloc(n_colors * n_channels * sizeof(float));
    float *h_new_centroids =
        (float *)malloc(n_colors * n_channels * sizeof(float));

    CHECK(cudaMemcpy(h_old_centroids, d_centroids,
                     n_colors * n_channels * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_new_centroids, d_new_centroids,
                     n_colors * n_channels * sizeof(float),
                     cudaMemcpyDeviceToHost));

    float centroid_shift = 0.0f;
    for (int i = 0; i < n_colors * n_channels; i++) {
      float diff = h_new_centroids[i] - h_old_centroids[i];
      centroid_shift += diff * diff;
    }

    free(h_old_centroids);
    free(h_new_centroids);

    // Swap centroid pointers
    float *temp = d_centroids;
    d_centroids = d_new_centroids;
    d_new_centroids = temp;

    if (centroid_shift < tolerance) {
      printf("Converged after %d iterations\n", iter + 1);
      break;
    }
  }

  // Final assignment
  if (use_shared) {
    assign_clusters_shared_kernel<<<blocks, THREADS_PER_BLOCK,
                                    shared_mem_size>>>(
        d_pixels, n_pixels, n_channels, d_centroids, n_colors, d_labels);
  } else {
    assign_clusters_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_pixels, n_pixels, n_channels, d_centroids, n_colors, d_labels);
  }
  cudaDeviceSynchronize();

  // Calculate inertia
  calculate_inertia_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      d_pixels, n_pixels, n_channels, d_centroids, d_labels, d_partial_inertia);
  cudaDeviceSynchronize();

  // Reduce partial inertia on host
  double *h_partial_inertia = (double *)malloc(blocks * sizeof(double));
  CHECK(cudaMemcpy(h_partial_inertia, d_partial_inertia,
                   blocks * sizeof(double), cudaMemcpyDeviceToHost));

  double total_inertia = 0.0;
  for (int i = 0; i < blocks; i++) {
    total_inertia += h_partial_inertia[i];
  }
  total_inertia /= n_pixels;

  free(h_partial_inertia);

  // Copy results back to host
  CHECK(cudaMemcpy(centroids, d_centroids,
                   n_colors * n_channels * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(labels, d_labels, n_pixels * sizeof(int),
                   cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(d_pixels);
  cudaFree(d_centroids);
  cudaFree(d_new_centroids);
  cudaFree(d_labels);
  cudaFree(d_cluster_sums);
  cudaFree(d_cluster_counts);
  cudaFree(d_partial_inertia);
  free(h_pixels);

  return total_inertia;
}
