//
// Created by pi on 12/3/21.
//

#ifndef HPCFINALCUDA_KMEANS_CUH
#define HPCFINALCUDA_KMEANS_CUH
#include "malloc.h"

#define TPB 32
#define COLUMNS 6
#define CLUSTERS 6

/*
 * Our model is defined as a struct and contains all the values needed for
 * clustering.
 */
struct KMeans{
    int *d_no_clusters; // Total number of clusters
    float *centroids;   // Centroids for each cluster.
    int *d_rows;        // Total number of data rows.
    int *d_columns;     // Total number of columns of data.
    float *data;        // Data to be clustered, (Dimension: rows * columns)
    int *data_clusters; // Clusters for each row in data. Dimension: (rows)
    int *cluster_size;  // Total number of data_elements in each cluster for each iteration.
    float  *cluster_sum;// Total sum of the data_points in each cluster. Used to compute centroid average.

    int columns;
    int rows;
    int no_clusters;
};

void printCentroids(struct KMeans *model);
void printClusterCount(struct KMeans *model);
void init_model(struct KMeans *cluster);
void writeToCSV(struct KMeans *model, char *filename);

__global__
void fit_cuda(struct KMeans *model);
#endif //HPCFINALCUDA_KMEANS_CUH
