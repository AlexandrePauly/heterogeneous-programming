// Bibliothèques importées
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "lodepng.h"  // Bibliothèque pour le chargement et la sauvegarde des images PNG
 
#define SIGMA_S 2.0   // Paramètre de lissage spatial (contrôle la diffusion dans l'espace)
#define SIGMA_R 50.0  // Paramètre de préservation des contours (contrôle la diffusion en intensité)
#define KERNEL_SIZE 5 // Taille du noyau du filtre bilatéral
 
// Structure pour stocker une image
typedef struct {
    unsigned char *image;   // Pointeur vers les pixels de l'image
    unsigned width, height; // Dimensions de l'image
} Image;
 
// Fonction pour charger une image PNG en mémoire
void load_image(const char *filename, Image *img) {
    unsigned error = lodepng_decode32_file(&img->image, &img->width, &img->height, filename);
    if (error) {
        printf("Erreur lors du chargement de l'image: %s\n", lodepng_error_text(error));
        exit(1);
    }
}
 
// Fonction pour sauvegarder une image PNG
void save_image(const char *filename, Image *img) {
    unsigned error = lodepng_encode32_file(filename, img->image, img->width, img->height);
    if (error) {
        printf("Erreur lors de l'enregistrement de l'image: %s\n", lodepng_error_text(error));
        exit(1);
    }
}
 
// Filtre bilatéral implémenté sur GPU
__global__ void bilateral_filter_cuda(unsigned char *d_input, unsigned char *d_output, int width, int height) {
    // Calcul de la position du thread dans l'image -> un thread est assigné à chaque pixel unique de l’image.
    // Thread permet : une exécution parallèle, l'optimisation des performances et la réactivité
    // blockIdx.x et blockIdx.y : Indice du bloc actuel
    // blockDim.x et blockDim.y : Nombre de threads par bloc
    // threadIdx.x et threadIdx.y : Indice du thread dans son bloc
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
 
    // Vérifier si le thread est en dehors de l'image
    if (x >= width || y >= height) return;
 
    // Calcul de l'index du pixel actuel dans l'image (RGBA, donc 4 canaux)
    int idx = 4 * (y * width + x);
    float sum_r = 0, sum_g = 0, sum_b = 0, norm_factor = 0;
    float sigma_s2 = 2.0 * SIGMA_S * SIGMA_S; // Variance du filtre spatial
    float sigma_r2 = 2.0 * SIGMA_R * SIGMA_R; // Variance du filtre en intensité
    int half_size = KERNEL_SIZE / 2;          // Moitié de la taille du noyau
 
    // Récupération des valeurs des canaux du pixel central
    unsigned char r = d_input[idx], g = d_input[idx+1], b = d_input[idx+2];
 
    // Parcours de la fenêtre de voisinage autour du pixel actuel
    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            int yy = y + i;
            int xx = x + j;

            // Si le voisin est dans l'image
            // C'est pour cette raison, que le lissage est meilleur au centre de l'image que sur les bords.
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                int neighbor_idx = 4 * (yy * width + xx);
                unsigned char nr = d_input[neighbor_idx], ng = d_input[neighbor_idx+1], nb = d_input[neighbor_idx+2];
                
                // Calcul des poids spatial et d'intensité
                float intensity_diff = (r - nr) * (r - nr) + (g - ng) * (g - ng) + (b - nb) * (b - nb);
                float range_weight = expf(-intensity_diff / sigma_r2);
                float spatial_weight = expf(-(i * i + j * j) / sigma_s2);
                float weight = spatial_weight * range_weight;
                
                // Accumulation des valeurs pondérées
                sum_r += weight * nr;
                sum_g += weight * ng;
                sum_b += weight * nb;
                norm_factor += weight;
            }
        }
    }
 
    // Mise à jour des valeurs filtrées
    d_output[idx] = (unsigned char)(sum_r / norm_factor);
    d_output[idx+1] = (unsigned char)(sum_g / norm_factor);
    d_output[idx+2] = (unsigned char)(sum_b / norm_factor);
    d_output[idx+3] = d_input[idx+3]; // Conserver l'alpha
}
 
// Fonction de gestion du filtre bilatéral en CUDA
void bilateral_filter(Image *img) {
    // Initialisation des variables
    int width = img->width;   // Width de l'image
    int height = img->height; // Height de l'image
    size_t img_size = width * height * 4 * sizeof(unsigned char); // Dimensions de l'image
    unsigned char *d_input, *d_output;
    
    // Allocation mémoire sur GPU
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMemcpy(d_input, img->image, img_size, cudaMemcpyHostToDevice);
 
    // Définition des tailles de bloc et de grille
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Lancement du noyau CUDA
    bilateral_filter_cuda<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Attente de la fin de l'exécution GPU
    
    // Récupération des résultats
    cudaMemcpy(img->image, d_output, img_size, cudaMemcpyDeviceToHost);
    
    // Libération de la mémoire GPU
    cudaFree(d_input);
    cudaFree(d_output);
}
 
// Programme principal
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }
 
    Image img;
    
    // Chargement de l'image
    load_image(argv[1], &img);
    
    // Mesure du temps d'exécution sur GPU
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);  // Début du chronomètre
    
    // Application du filtre bilatéral
    bilateral_filter(&img);
    
    cudaEventRecord(stop, 0);   // Fin du chronomètre
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Temps total d'exécution du filtre bilatéral : %f ms\n", elapsedTime);
    
    // Sauvegarde de l'image traitée
    save_image(argv[2], &img);
    
    // Libération de la mémoire
    free(img.image);
    
    return 0;
}