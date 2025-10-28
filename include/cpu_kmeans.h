#ifndef CPU_KMEANS_H
#define CPU_KMEANS_H

// Functions
#ifdef __cplusplus
extern "C" {
#endif

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
                           int *labels, int max_iter, float tolerance);

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
                    int n_colors, float *centroids);

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
                     const float *centroids, int n_colors, int *labels);

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
                         const float *centroids, const int *labels);
#ifdef __cplusplus
}
#endif

#endif // CPU_KMEANS_H
