# CUDA Image Color Segmentation

A high-performance (kinda) image color segmentation tool using K-means clustering, with both CPU and GPU (CUDA) implementations, demonstrating the power of GPU acceleration for image processing tasks.

## Overview

This tool performs K-means clustering on image pixels to segment images by dominant colors. It supports both JPEG and PNG formats.

## Features

- **Dual Implementation**: Choose between CPU and GPU (CUDA) processing
- **K-means++ Initialization**: Smart centroid initialization for better convergence
- **Optimized CUDA Kernels**: 
  - Shared memory optimization for small centroid sets
  - Coalesced memory access patterns
  - Parallel reduction for inertia calculation
  - Atomic operations for efficient accumulation
- **Flexible Configuration**: Specify number of color clusters (2-255)
- **Multiple Image Formats**: Supports JPEG and PNG images ("multiple" means 2)
- **Magic Number Validation**: Validates image files by header, not extension

## Requirements

### Software
- CUDA Toolkit 
- GCC/G++ compiler 

### Hardware
- NVIDIA GPU (for GPU acceleration)
- Sufficient RAM for image processing (varies by image size)

## Project Structure

```
.
├── include/
│   ├── color_segmentation.h
│   ├── cpu_kmeans.h
│   └── gpu_kmeans.h
├── src/
│   ├── color_segmentation.cu
│   ├── cpu_kmeans.c
│   └── gpu_kmeans.cu
└── README.md
```

## Building the Project

Just `make` it, and `make clean` removes the created directories.

## Usage

### Basic Syntax

```bash
color_segment <image_path> [num_colors] [--cpu]
```

### Parameters

- `<image_path>`: Path to the input image (required)
- `[num_colors]`: Number of color clusters (optional, 2-255, default: 4)
- `[--cpu]`: Use CPU implementation (optional)

## Output

The program generates:
1. **Console output**: 
   - Device information (GPU name, compute capability)
   - Dominant colors in RGB format
   - Final inertia value (clustering quality metric)
   - Convergence information

2. **Output image**: 
   - Saved as `<original_name>_clustered.png`
   - Each pixel colored by its assigned cluster centroid
   - Kinda pretty

## Algorithm Details

### K-means Clustering

The algorithm iteratively:
1. Assigns each pixel to the nearest centroid (color)
2. Recalculates centroids as the mean of assigned pixels
3. Repeats until convergence or maximum iterations

### K-means++ Initialization

Initial centroids are selected using the K-means++ algorithm, which:
- Chooses the first centroid randomly
- Selects subsequent centroids with probability proportional to their squared distance from existing centroids
- Provides better initial conditions than random selection

### GPU Optimizations

#### Shared Memory
- Centroids are cached in shared memory when small enough (< 4096 floats, so almost always)
- Reduces global memory bandwidth requirements

#### Parallel Reduction
- Inertia calculation uses tree-based reduction in shared memory
- Minimizes thread divergence and synchronization overhead

#### Coalesced Memory Access
- Array layouts designed for consecutive thread access patterns
- Maximizes memory throughput

#### Atomic Operations
- Safe parallel accumulation of cluster sums and counts
- Enables efficient centroid updates across all threads

## Performance

Expected speedup depends on:
- Image size (larger images benefit more from GPU)
- Number of clusters (more clusters/colors benefits more form GPU)
- GPU compute capability
- Number of iterations required for convergence

## References

- [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [K-means++ Initialization](https://en.wikipedia.org/wiki/K-means%2B%2B)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [STB Image Libraries](https://github.com/nothings/stb)

## Acknowledgments

- STB libraries by Sean Barrett for image I/O
- NVIDIA CUDA team for the CUDA toolkit and documentation
