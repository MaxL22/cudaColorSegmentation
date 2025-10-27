#ifndef CPU_KMEANS_H
#define CPU_KMEANS_H

// Functions
#ifdef __cplusplus
extern "C" {
#endif

double kmeans_image_colors(const unsigned char *data, int n_pixels,
                           int n_channels, int n_colors, float *centroids,
                           int *labels, int max_iter, float tolerance);
void kmeans_pp_init(const float *pixels, int n_pixels, int n_channels,
                    int n_colors, float *centroids);
void assign_clusters(const float *pixels, int n_pixels, int n_channels,
                     const float *centroids, int n_colors, int *labels);
double calculate_inertia(const float *pixels, int n_pixels, int n_channels,
                         const float *centroids, const int *labels);
#ifdef __cplusplus
}
#endif

#endif // CPU_KMEANS_H
