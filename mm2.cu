#include <stdio.h>
#include <stdlib.h>

// Funkcja kernela CUDA do mnożenia macierzy
__global__ void matrixMultiply(int *a, int *b, int *c, int width) {
    // Pobierz identyfikator bloku i wątku
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Zmienna do przechowywania wyniku mnożenia
    int sum = 0;

    // Mnożenie elementów i sumowanie wyniku
    for (int i = 0; i < width; ++i) {
        sum += a[row * width + i] * b[i * width + col];
    }

    // Zapisz wynik w macierzy wynikowej
    c[row * width + col] = sum;
}

// Funkcja do wyświetlania macierzy
void printMatrix(int *matrix, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d\t", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Rozmiar macierzy (kwadratowej w tym przypadku)
    int width = 16; // Zmniejszamy rozmiar do lepszego wyświetlania
    int size = width * width * sizeof(int);

    // Alokuje pamięć dla macierzy na CPU
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    // Inicjalizacja macierzy
    for (int i = 0; i < width * width; ++i) {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    // Alokuje pamięć dla macierzy na GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Kopiuje dane z CPU do GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Definiuje rozmiar bloku i siatkę wątków CUDA
    dim3 blockSize(2, 2); // Mniejszy rozmiar bloku do lepszego wyświetlania
    dim3 gridSize(width / blockSize.x, width / blockSize.y);

    // Wywołuje kernel CUDA do mnożenia macierzy
    matrixMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);

    // Kopiuje wynik z GPU do CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Wyświetla macierze
    printf("Matrix A:\n");
    printMatrix(h_a, width);

    printf("Matrix B:\n");
    printMatrix(h_b, width);

    printf("Result Matrix C:\n");
    printMatrix(h_c, width);

    // Zwolnij pamięć
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
