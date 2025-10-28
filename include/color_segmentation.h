#ifndef COLOR_SEGMENTATION_H
#define COLOR_SEGMENTATION_H

#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>

// Constants
#define COLOR_NUMBER 4
#define S_ERROR "[ERROR]"
#define S_WARNING "[WARNING]"
#define MAX_ITER 200
#define EPSILON 1e-4

// Data structs
typedef struct {
  int width;
  int height;
  int channels;
  char *filename;
  unsigned char *data;
} LoadedImage;

// Functions
/**
 * is_valid_image: Validates if a file is a valid PNG or JPEG image
 * @filename: Path to the image file to validate
 *
 * Checks the file's magic numbers to determine if it's
 * a valid PNG or JPEG image. Does NOT rely on file extension.
 *
 * Returns: true if file is valid PNG or JPEG, false otherwise
 */
void print_help(const char *program_name);

/**
 * is_valid_image: Validates if a file is a valid PNG or JPEG image
 * @filename: Path to the image file to validate
 *
 * Checks the file's magic numbers to determine if it's
 * a valid PNG or JPEG image. Does NOT rely on file extension.
 *
 * Returns: true if file is valid PNG or JPEG, false otherwise
 */
bool is_valid_image(const char *filename);

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
unsigned char *load_image(char *dir, LoadedImage *img);

/**
 * cleanup: free all pointers used
 * @centroids: array of centroids
 * @labels: array of labels
 * @img_data: data of an image, loaded with stbi_load()
 */
void cleanup(float *centroids, int *labels, unsigned char *img_data);

/**
 * dump_image: write a file with the image, as divided in clusters
 * @centroids: values of the centroids (colors to assign)
 * @labels: label for each pixel
 */
void dump_image(const float *centroids, const int *labels, LoadedImage *img);

/**
 * name_changer: Gives back a new name with "_clustered.png" at the end
 * @filepath: original filepath
 *
 * Return: the new name, it's just a string, NULL if failed
 */
char *name_changer(const char *filepath);

#endif // COLOR_SEGMENTATION_H
