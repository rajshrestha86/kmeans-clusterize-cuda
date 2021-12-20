//
// Created by pi on 12/3/21.
//

#include <stdio.h>
#include "kmeans.cuh"
#include <malloc.h>
#include <stdlib.h>
#include <math.h>

#define SEED 100

void randomCentroids(struct KMeans *model) {
    /*
     * Initialize our model with initial centroids.
     * Just generate the random number between 0 and number of rows and assign the
     * centroids as initial centroids.
     */
    float *data = model->data;
    int rows = model->rows;
    int columns = model -> columns;
    int clusters = model->no_clusters;
    for (int cluster = 0; cluster < clusters; cluster++) {
        // Between 0 & 1
        float random_number = ((float ) rand() /  (float ) RAND_MAX);
        int random_row =  (int) (random_number * rows);
        printf("\nRandom index number: %f, %d\n", random_number, random_row);
        for (int i = 0; i < columns; i++) {
            printf("%f,", data[random_row*columns + i]);
            model -> centroids[columns * cluster + i] = data[random_row * columns + i];
        }
        printf("\n");
    }
}

__device__
float eucledianDist(struct KMeans *model, int pt2_index, int cluster) {
    /*
     * This function computes the eucledian distance for two points.
     * Given, (x, y) and (a, b)
     * It returns:
     *      (x-a)*(x-a) + (y-b)*(y-b)
     */
    float dist = 0;
    float *pt1 = model -> centroids;
    for (int i = 0; i < model -> columns; i++){
        int data_index = COLUMNS * pt2_index + i;
        dist += (pt1[cluster * COLUMNS + i] - model -> data[data_index]) * (pt1[cluster * COLUMNS + i] - model -> data[data_index]);
    }
    return sqrtf(dist);
}


void init_model(struct KMeans *cluster) {
    /*
     * This function assigns the initial centroids to out model.
     * The initial centroids are computed by assigning random data points as initial centroids.
     */
    randomCentroids(cluster);
}



__device__
void update_centroids_gpu(struct KMeans *model) {
    /*
     * This function computes the total_centroid sum of data points inside the cluster
     * and the total number of data points inside the cluster. This information is used on
     * host to compute the centroid average.
     *
     * In order to reduce access to global memory we load all the elements to a tile and compute the average.
     * In this code we parallelize the access to global memory data, and using just a thread with rank 0, to
     * compute the sum of centroids and the size of clusters.
     */
    __shared__ float s_data[TPB * COLUMNS];
    __shared__ int s_data_clusters[TPB];

    int data_index = blockDim.x * blockIdx.x + threadIdx.x;
    const int s_idx = threadIdx.x;
    /*
     * The number of threads could be less than data so we repeat until loop condition is satisfied.
     */
    while (data_index < model -> rows){

        /*
         * Load data to shared memory.
         */
        s_data_clusters[s_idx] = model->data_clusters[data_index];
        for(int i = 0; i < COLUMNS; i++){
            s_data[s_idx * COLUMNS + i] = model->data[data_index * COLUMNS + i];
        }
        // Wait for all threads in a block to load the data.
        __syncthreads();
        /*
         * At this point all the required data has been loaded in shared memory.
         * Now we compute the sum of the centroids and calculate the size of each cluster.
         */
        if (s_idx == 0) {
            float cluster_datapoint_sums[CLUSTERS * COLUMNS] = {0};
            int cluster_sizes[CLUSTERS] = {0};
            // Add all the elements to centroid and increase cluster size accordingly.
            for (int rank = 0; rank < blockDim.x; rank++) {
                int cluster = s_data_clusters[rank];
                for(int i = 0; i < COLUMNS; i++){
                    cluster_datapoint_sums[cluster * COLUMNS + i] += s_data[rank * COLUMNS + i];
                }
                cluster_sizes[cluster] += 1;
            }

            // After local sum has been computed, its time to add them to global variables.
            for (int k = 0; k < model->no_clusters; k++) {
                for(int c = 0; c < COLUMNS; c++){
                    atomicAdd(&model->cluster_sum[k * COLUMNS + c], cluster_datapoint_sums[k * COLUMNS + c]);
                }
                atomicAdd(&model->cluster_size[k], cluster_sizes[k]);
            }
        }
        __syncthreads();
        data_index += gridDim.x * blockDim.x;
    }
}

void printCentroids(struct KMeans *model){
    /*
     * This function just prints the centroid of a KMeans model.
     * Used for debugging.
     */

    printf("\n printing centroids: \n");
    printf("Total clusters: %d\n", model -> no_clusters);
    for (int cluster = 0; cluster < model -> no_clusters; cluster++) {
        for (int i = 0; i < model -> columns; i++) {
            float point = model -> centroids[model -> columns * cluster + i];
            printf("%f, ", point);
        }
        printf("\n");
    }
}

void printClusterCount(struct KMeans *model){
    /*
     * This function just prints the cluster assignment of data.
     * Used to see results after completing iteration and debugging.
     */
    int cluster_count[model -> no_clusters];
    for(int i = 0; i < model -> no_clusters; i++)
        cluster_count[i] = 0;

    for(int row = 0; row < model -> rows; row++){
        for (int k = 0; k < model -> no_clusters; k++){
            if(model -> data_clusters[row] == k){
                cluster_count[k] = cluster_count[k] + 1;
                break;
            }
        }
    }

    for(int i = 0; i < model -> no_clusters; i++)
        printf("\nCount for cluster: %d = %d", i, cluster_count[i]);
}

__global__
void fit_cuda(struct KMeans *model) {
    /*
     * This function fits the KMeans model.
     * For each data point assign the nearest cluster by looking at the eucledian distance
     * and re-compute the centroids of each cluster.
     */

    // Just dummy variables for code-compatibility with Serial code.
    model -> no_clusters = *model -> d_no_clusters;
    model -> rows = *model -> d_rows;
    model -> columns = *model -> d_columns;

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    while (index < model->rows) {
        // Using eucledian distance as Distance Metric.
        float min_dist =  eucledianDist( model, index, 0);;
        model -> data_clusters[index] = 0;
        for (int k = 1; k < model->no_clusters; k++) {
            float dist = eucledianDist(model, index, k);
            // Assign the row to a nearest cluster.
            if(dist <= min_dist){
                min_dist = dist;
                model -> data_clusters[index] = k;
            }
        }
        index += (int) blockDim.x * gridDim.x;
    }
    /*
     * Waiting for all the threads in a block to complete their part and
     * after that we will update the centroids of current clusters.
     */
    __syncthreads();
    update_centroids_gpu(model);
}



void writeToCSV(struct KMeans *model, char *filename) {
    /*
     * Write results of clustered model to file in filename.
     * This function creates (model -> rows) number of rows with columns: (model -> columns)
     * There is also a last column which is the cluster assigned to each data_point.
     * This file is used to create a visualization for cluster.
     */
    int row = model -> rows;
    int columns = model -> columns;
    float *data = model -> data;
    FILE *fp;
    int row_index = 0;
    fp = fopen(filename, "a");
    while (row_index < row) {
        int column = 0;
        while(column < columns){
            fprintf(fp, "%f,", data[row_index * columns + column]);
            column++;
        }
        fprintf(fp, "%d", model -> data_clusters[row_index]);
        fprintf(fp, "\n");
        row_index++;
    }
    fclose(fp);
}