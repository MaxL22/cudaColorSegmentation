#include "../include/cpu_kmeans.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * kmeans_image_colors - Performs k-means clustering on image pixel colors
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
 * Return: Final inertia value (sum of squared distances)
 */
double kmeans_image_colors(const unsigned char *data, int n_pixels,
                           int n_channels, int n_colors, float *centroids,
                           int *labels, int max_iter, float tolerance) {
  // Allocate space for the pixels
  float *pixels = malloc(n_pixels * n_channels * sizeof(float));
  // How many pixels in each cluster
  int *cluster_sizes = malloc(n_colors * sizeof(int));
  // Centroids, to be updated each iteration
  float *new_centroids = malloc(n_colors * n_channels * sizeof(float));

  // Convert byte data to float
  for (int i = 0; i < n_pixels * n_channels; i++) {
    pixels[i] = (float)data[i];
  }

  // Initialize centroids using k-means++ algorithm
  kmeans_pp_init(pixels, n_pixels, n_channels, n_colors, centroids);

  // Main k-means loop
  for (int iter = 0; iter < max_iter; iter++) {
    // Assign each pixel to nearest centroid
    assign_clusters(pixels, n_pixels, n_channels, centroids, n_colors, labels);

    // Calculate new centroids
    memset(new_centroids, 0, n_colors * n_channels * sizeof(float));
    memset(cluster_sizes, 0, n_colors * sizeof(int));

    // For each pixel, increment the size of the cluster to which it belongs
    for (int i = 0; i < n_pixels; i++) {
      cluster_sizes[labels[i]]++;
      // Then sum the value of all centroids
      for (int c = 0; c < n_channels; c++) {
        new_centroids[labels[i] * n_channels + c] += pixels[i * n_channels + c];
      }
    }
    // Average to get the new centroids
    for (int k = 0; k < n_colors; k++) {
      if (cluster_sizes[k] > 0) {
        for (int c = 0; c < n_channels; c++) {
          new_centroids[k * n_channels + c] /= cluster_sizes[k];
        }
      } else {
        // Keep the old centroid value for empty clusters
        for (int c = 0; c < n_channels; c++) {
          new_centroids[k * n_channels + c] = centroids[k * n_channels + c];
        }
      }
    }

    // Check for convergence
    float centroid_shift = 0.0f;
    for (int i = 0; i < n_colors * n_channels; i++) {
      float diff = new_centroids[i] - centroids[i];
      centroid_shift += diff * diff;
    }

    memcpy(centroids, new_centroids, n_colors * n_channels * sizeof(float));

    if (centroid_shift < tolerance) {
      printf("Converged after %d iterations\n", iter + 1);
      break;
    }
  }

  // Final assignment and calculate inertia
  assign_clusters(pixels, n_pixels, n_channels, centroids, n_colors, labels);
  double inertia =
      calculate_inertia(pixels, n_pixels, n_channels, centroids, labels);

  free(pixels);
  free(cluster_sizes);
  free(new_centroids);

  return inertia;
}

/**
 * kmeans_pp_init: Initializes centroids using k-means++ algorithm (taken
 * from the pseudo-code on the wikipedia page)
 * https://en.wikipedia.org/wiki/K-means%2B%2B
 * @pixels: Array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @n_colors: Number of centroids to initialize
 * @centroids: Output array of shape (n_colors * n_channels) for initial
 * centroids
 */
void kmeans_pp_init(const float *pixels, int n_pixels, int n_channels,
                    int n_colors, float *centroids) {
  float *distances = malloc(n_pixels * sizeof(float));

  // Choose first centroid randomly
  int first_idx = rand() % n_pixels;
  for (int c = 0; c < n_channels; c++) {
    centroids[c] = pixels[first_idx * n_channels + c];
  }

  // Choose remaining centroids
  for (int k = 1; k < n_colors; k++) {
    // Calculate minimum distance to existing centroids
    for (int i = 0; i < n_pixels; i++) {
      float min_dist = FLT_MAX;
      for (int j = 0; j < k; j++) {
        float dist = 0.0f;
        for (int c = 0; c < n_channels; c++) {
          float diff =
              pixels[i * n_channels + c] - centroids[j * n_channels + c];
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
      centroids[k * n_channels + c] = pixels[chosen_idx * n_channels + c];
    }
  }

  free(distances);
}

/**
 * assign_clusters: Assigns each pixel to the nearest centroid
 * @pixels: Array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @centroids: Array of shape (n_colors * n_channels) containing cluster centers
 * @n_colors: Number of clusters
 * @labels: Output array of shape (n_pixels) with cluster indices
 */
void assign_clusters(const float *pixels, int n_pixels, int n_channels,
                     const float *centroids, int n_colors, int *labels) {
  for (int i = 0; i < n_pixels; i++) {
    float min_dist = FLT_MAX;
    int best_cluster = 0;

    // For each pixel find the minimum distance from a centroid, then set the
    // label
    for (int k = 0; k < n_colors; k++) {
      float dist = 0.0f;
      for (int c = 0; c < n_channels; c++) {
        float diff = pixels[i * n_channels + c] - centroids[k * n_channels + c];
        dist += diff * diff;
      }

      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = k;
      }
    }

    labels[i] = best_cluster;
  }
}

/**
 * calculate_inertia: Calculates sum of squared distances from pixels to
 * centroids
 * @pixels: Array of shape (n_pixels * n_channels) containing pixel data
 * @n_pixels: Total number of pixels
 * @n_channels: Number of channels per pixel
 * @centroids: Array of shape (n_colors * n_channels) containing cluster centers
 * @labels: Array of shape (n_pixels) with cluster assignments
 *
 * Return: Total inertia as a double
 */
double calculate_inertia(const float *pixels, int n_pixels, int n_channels,
                         const float *centroids, const int *labels) {
  double inertia = 0.0;

  for (int i = 0; i < n_pixels; i++) {
    int cluster = labels[i];
    for (int c = 0; c < n_channels; c++) {
      double diff =
          pixels[i * n_channels + c] - centroids[cluster * n_channels + c];
      inertia += diff * diff;
    }
  }

  return inertia / n_pixels;
}
