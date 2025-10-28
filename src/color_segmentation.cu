#include "../include/color_segmentation.h"
#include "../include/cpu_kmeans.h"
#include "../include/gpu_kmeans.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

int main(int argc, char *argv[]) {
  // Load the image
  // Parameters should be: image path, color count, --cpu

  // Check number of parameters
  if (argc < 2 || argc > 4) {
    fprintf(stderr, "%s Wrong number of parameters\n", S_ERROR);
    print_help(argv[0]);
    return 1;
  }

  // Parse params, I know there's a library, there's not too much to parse tho
  char *dir = argv[1];
  bool use_gpu = true;
  int num_colors = COLOR_NUMBER;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--cpu") == 0) {
      use_gpu = false;
    } else {
      num_colors = atoi(argv[i]);
      if (num_colors < 0 || num_colors > 256) {
        fprintf(stderr,
                "%s The number of colors should be between 2 and 255, falling "
                "back to default (%d) \n",
                S_WARNING, COLOR_NUMBER);
        num_colors = COLOR_NUMBER;
      }
    }
  }

  // Initialize CUDA
  int device_count;
  CHECK_WITH_CODE(cudaGetDeviceCount(&device_count), 2);

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Using device: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);

  // Check file
  if (!is_valid_image(dir)) {
    fprintf(stderr, "%s Not a valid JPEG or PNG image \n", S_ERROR);
    return 2;
  }

  // Load image
  LoadedImage *img = (LoadedImage *)malloc(sizeof(LoadedImage));
  img->data = load_image(dir, img);
  if (img->data == NULL) {
    fprintf(stderr, "%s Can't load image \n", S_ERROR);
    return 3;
  }

  // img->data is a continguous array of channels*number_of_pixels bytes,
  // e.g., for RGB it's: [R G B R G B ...]

  // Allocate output arrays
  float *centroids =
      (float *)malloc(num_colors * img->channels * sizeof(float));
  int n_pixels = img->height * img->width;
  int *labels = (int *)malloc(n_pixels * sizeof(int));

  // Perform k-means clustering
  double inertia;
  if (use_gpu) {
    inertia =
        gpu_kmeans_image_colors(img->data, n_pixels, img->channels, num_colors,
                                centroids, labels, MAX_ITER, EPSILON);
  } else {
    inertia =
        kmeans_image_colors(img->data, n_pixels, img->channels, num_colors,
                            centroids, labels, MAX_ITER, EPSILON);
  }

  printf("Dominant colors (RGB):\n");
  for (int i = 0; i < num_colors; i++) {
    printf("  Color %d: (%.1f, %.1f, %.1f)\n", i + 1,
           centroids[i * img->channels + 0], centroids[i * img->channels + 1],
           centroids[i * img->channels + 2]);
  }
  printf("Inertia: %.2f\n", inertia);

  // Dump new image to file
  dump_image(centroids, labels, img);

  // Clean everything
  cleanup(centroids, labels, img->data);

  return 0;
}

/**
 * dump_image: write a file with the image, as divided in clusters
 * @centroids: values of the centroids (colors to assign)
 * @labels: label for each pixel
 */
void dump_image(const float *centroids, const int *labels, LoadedImage *img) {
  int n_pixel = img->width * img->height;
  unsigned char *new_data =
      (unsigned char *)malloc(n_pixel * img->channels * sizeof(unsigned char));
  for (int i = 0; i < n_pixel; i++) {
    for (int j = 0; j < img->channels; j++) {
      new_data[i * img->channels + j] =
          centroids[labels[i] * img->channels + j];
    }
  }
  stbi_write_png(name_changer(img->filename), img->width, img->height,
                 img->channels, new_data, img->width * img->channels);

  free(new_data);
}

/**
 * name_changer: Gives back a new name with "_clustered.png" at the end
 * @filepath: original filepath
 *
 * Return: the new name, it's just a string, NULL if failed
 */
char *name_changer(const char *filepath) {
  const char *dot = strrchr(filepath, '.');
  size_t base_len = dot ? (size_t)(dot - filepath) : strlen(filepath);

  const char *suffix = "_clustered.png";
  size_t total_len = base_len + strlen(suffix);

  char *new_name = (char *)malloc(total_len + 1);
  if (!new_name)
    return NULL;

  memcpy(new_name, filepath, base_len);
  strcpy(new_name + base_len, suffix);
  return new_name;
}

/**
 * load_image: Loads an image file into memory
 * @img: LoadedImage struct with only filename initialized
 *
 * Loads an image file and returns a pointer to the pixel data.
 * The caller is responsible for freeing the returned data using
 * stbi_image_free().
 *
 * Return: Pointer to unsigned char array containing pixel data, or NULL on
 * failure
 */
unsigned char *load_image(char *dir, LoadedImage *img) {
  img->filename = dir;
  unsigned char *data = stbi_load(img->filename, &(img->width), &(img->height),
                                  &(img->channels), 0);

  if (!data) {
    return NULL;
  }

  return data;
}

/**
 * is_valid_image: Validates if a file is a valid PNG or JPEG image
 * @filename: Path to the image file to validate
 *
 * Checks the file's magic numbers to determine if it's
 * a valid PNG or JPEG image. Does NOT rely on file extension.
 *
 * Returns: true if file is valid PNG or JPEG, false otherwise
 */
bool is_valid_image(const char *filename) {
  if (!filename)
    return false;
  FILE *file = fopen(filename, "rb");
  if (!file)
    return false;

  // Reads header of the function
  unsigned char header[8];
  size_t bytes_read = fread(header, 1, sizeof(header), file);
  fclose(file);
  // too smol
  if (bytes_read < 3)
    return false;

  // JPEG: FF D8 FF
  if (header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF)
    return true;

  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (bytes_read >= 8 && header[0] == 0x89 && header[1] == 0x50 &&
      header[2] == 0x4E && header[3] == 0x47 && header[4] == 0x0D &&
      header[5] == 0x0A && header[6] == 0x1A && header[7] == 0x0A)
    return true;

  return false;
}

/**
 * print_help: prints usage of the program
 * @program_name: name of the program, usually taken from argv
 */
void print_help(const char *program_name) {
  printf("Usage: %s <image path> [options]\n\n", program_name);
  printf("Performs k-means clustering on the colors of a given image.\n\n");
  printf("Arguments:\n");
  printf("  <image path>       Path to the image (JPG or PNG)\n\n");
  printf("Options:\n");
  printf("  --cpu             Use CPU instead of GPU for computation\n");
  printf(
      "  <num_colors>      Number of color clusters (1-255, default: %d)\n\n",
      COLOR_NUMBER);
  printf("Examples:\n");
  printf("  %s ./images\n", program_name);
  printf("  %s ./images 4\n", program_name);
  printf("  %s ./images 8 --cpu\n", program_name);
}

/**
 * cleanup: free all pointers used
 * @centroids: array of centroids
 * @labels: array of labels
 * @img_data: data of an image, loaded with stbi_load()
 */
void cleanup(float *centroids, int *labels, unsigned char *img_data) {
  STBI_FREE(img_data);
  free(centroids);
  free(labels);
}
