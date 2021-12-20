//
// Created by pi on 12/5/21.
//
#include <stdio.h>
#include<stdio.h>
#include<string.h>
#include<stdbool.h>
#include"./lib/csvReader.cuh"
#include "lib/kmeans.cuh"
#include <time.h>
#define MAXCHAR 1000

int main(int argc, char *argv[]) {
    int clusters = CLUSTERS;
    char filename[] = "data/genres_v2.csv";
    struct CSVFile csvData = readCsv(filename);
    time_t t;
    t = time(NULL);
    /*
     * Initialization of KMeans model on host...
     * We wll first load data from CSV, and create necessary variables required for clustering.
     */
    struct KMeans model = {
            .d_no_clusters = (int *) malloc(sizeof(int)),
            .centroids = (float *) malloc(sizeof(float) * csvData.columns * clusters),
            .d_rows = (int *) malloc(sizeof(int)),
            .d_columns = (int *) malloc(sizeof(int)),
            .data = csvData.data,
            .data_clusters = (int *) malloc(sizeof(int) * csvData.rows),
            .cluster_size = (int *) malloc(sizeof(int) * clusters),
            .cluster_sum = (float *) malloc(sizeof(float ) * clusters * COLUMNS)
    };

    *(model.d_no_clusters) = clusters;
    *(model.d_rows) = csvData.rows;
    *(model.d_columns) = csvData.columns;
    // Initalize sum variables to 0.
    for (int k = 0; k < clusters; k++){
        model.cluster_size[k] = 0;
        for(int c = 0; c < COLUMNS; c++)
            model.cluster_sum[k * COLUMNS + c] = 0;
    }

    model.no_clusters = *model.d_no_clusters;
    model.columns = *model.d_columns;
    model.rows = *model.d_rows;
    init_model(&model);

    int *d_no_clusters, *d_rows, *d_columns, *d_data_clusters, *d_cluster_size;
    float *d_centroids, *d_data, *d_cluster_sum;

    /*
     * Create respective cuda variables and allocate them in CUDA device.
     */
    cudaMalloc(&d_no_clusters, sizeof(int));
    cudaMalloc(&d_rows, sizeof(int));
    cudaMalloc(&d_columns, sizeof(int));
    cudaMalloc(&d_data_clusters, sizeof(int )  * csvData.rows);
    cudaMalloc(&d_data, sizeof(float) * csvData.rows * csvData.columns);
    cudaMalloc(&d_centroids, sizeof(float) * clusters * csvData.columns);
    cudaMalloc(&d_cluster_sum, sizeof(float) * clusters * csvData.columns);
    cudaMalloc(&d_cluster_size, sizeof(int) * clusters);


    // Now copy data to cuda
    cudaMemcpy(d_no_clusters, &clusters, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, &csvData.rows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &csvData.columns, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, csvData.data, sizeof(int) * csvData.rows * csvData.columns, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, model.centroids, sizeof(float) * clusters * csvData.columns, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_size, model.cluster_size, sizeof(int ) * clusters, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_sum, model.cluster_sum, sizeof(float) * clusters * csvData.columns, cudaMemcpyHostToDevice);

    /*
     * Prepare model to be copied to CUDA device.
     */
    KMeans *h_model = (KMeans *) malloc(sizeof(KMeans));
    h_model->d_no_clusters = d_no_clusters;
    h_model->d_rows = d_rows;
    h_model->d_columns = d_columns;
    h_model->data_clusters = d_data_clusters;
    h_model->data = d_data;

    int iteration = 300;
    KMeans *d_model;
    cudaMalloc((void **) &d_model, sizeof(KMeans));
    /*
     * Run kernel for 300 iteration.
     * Each iteration will parallelize the KMeans algorithm and centroid update method.
     */
    while(iteration >= 0){

        cudaMemcpy(d_centroids, model.centroids, sizeof(float) * clusters * csvData.columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_sum, model.cluster_sum, sizeof(float) * clusters * csvData.columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_size, model.cluster_size, sizeof(int ) * clusters, cudaMemcpyHostToDevice);

        h_model->cluster_size = d_cluster_size;
        h_model->centroids = d_centroids;
        h_model -> cluster_sum = d_cluster_sum;

        cudaMemcpy(d_model, h_model, sizeof(KMeans), cudaMemcpyHostToDevice);

        fit_cuda<<<1, TPB>>>(d_model);

        cudaMemcpy(model.cluster_sum, d_cluster_sum, sizeof(float) * clusters * COLUMNS, cudaMemcpyDeviceToHost);
        cudaMemcpy(model.centroids, d_centroids, sizeof(float) * clusters * COLUMNS, cudaMemcpyDeviceToHost);
        cudaMemcpy(model.cluster_size, d_cluster_size, sizeof(int) * CLUSTERS, cudaMemcpyDeviceToHost);
        for(int k =0; k < CLUSTERS; k++){
            for (int c = 0; c < COLUMNS; c++) {
                int index = k * COLUMNS + c;
                if(model.cluster_size[k] > 0){
                    model.centroids[index] = model.cluster_sum[index] / ((float) model.cluster_size[k]);
//                    model.centroids[index] = model.cluster_sum[index];
//                    model.centroids[index] = 10;
                }
                model.cluster_sum[index] = 0;
            }
            model.cluster_size[k] = 0;
        }
        iteration--;
    }
    cudaMemcpy(model.data_clusters, d_data_clusters, sizeof(int) * csvData.rows, cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n%s\n", cudaGetErrorString(error), cudaGetErrorName(error));
        exit(-1);
    }

    printf("\n After fitting.... ");
    printCentroids(&model);
    printClusterCount(&model);

    time_t end = time(NULL);
    double time_taken = difftime(end, t);
    printf("\n\nThe program took %f seconds to execute.\n\n", time_taken);
    // Result file.
    char resFile[] = "clustered_gpu.csv";
    writeToCSV(&model, resFile);

    // Free device memory.
    cudaFree(&d_no_clusters);
    cudaFree(&d_rows);
    cudaFree(&d_columns);
    cudaFree(&d_data_clusters);
    cudaFree(&d_data) ;
    cudaFree(&d_centroids);
    cudaFree(&d_cluster_sum);
    cudaFree(&d_cluster_size);
    return 0;

}