// ----------------------------------------------------------------------------
// Distância de edição paralelizado (solução completa)
//
// Estudante: Andrews Matheus de Oliveira
// Estudante: Guilherme Gonzaga de Andrade
// Estudante: Walter do Espirito Santo Souza Filho
//
// Para compilar: nvcc dist_par.cu -o dist_par
// Para executar: ./dist_par <nome arquivo entrada>
// ----------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int kTamBloco = 1024;


int divteto(int a, int b) {
	return a / b + (a % b != 0);
}

char *aloca_sequencia_host(int n) {
	char *seq = (char *) malloc((n + 1) * sizeof(char));

	if (seq == NULL) {
		printf("\nErro na alocação de estruturas\n");
		exit(EXIT_FAILURE);
	}

	return seq;
}

void ler_entrada(const char *nomearq, int *n, int *m, char **h_s, char **h_r) {
	// Abre arquivo de entrada
	FILE *arqEntrada = fopen(nomearq, "rt");

	if (arqEntrada == NULL) {
		printf("\nArquivo texto de entrada não encontrado\n");
		exit(EXIT_FAILURE);
	}

	// Lê tamanho das sequências s e r
	fscanf(arqEntrada, "%d %d", n, m);

	*h_s = aloca_sequencia_host(*n);
	*h_r = aloca_sequencia_host(*m);

	(*h_s)[0] = ' ';
	(*h_r)[0] = ' ';
	fscanf(arqEntrada, "%s", *h_s + 1);
	fscanf(arqEntrada, "%s", *h_r + 1);

	// Fecha arquivo de entrada
	fclose(arqEntrada);
}

void aloca_dev(int n, int m, char **d_s, char **d_r, int **d_dist) {
	if (cudaSuccess != cudaMalloc(d_s, (n + 1) * sizeof(char)) ||
	    cudaSuccess != cudaMalloc(d_r, (m + 1) * sizeof(char)) ||
	    cudaSuccess != cudaMalloc(d_dist, (n + 1) * (m + 1) * sizeof(int))) {
		printf("\nErro na alocação de estruturas\n");
		exit(EXIT_FAILURE);
	}
}

// Retorna o valor mínimo entre a, b e c
__device__
int min(int a, int b, int c) {
	return (a < b) ? min(a, c) : min(b, c);
}

__device__
void prepara_matriz(int n, int m, int *d_dist) {
	int idThread = blockDim.x * blockIdx.x + threadIdx.x;
	// Inicializa primeira linha da matriz
	for (int i = idThread; i <= m; i += gridDim.x * blockDim.x) {
		d_dist[i] = i;
	}

	// Inicializa primeira coluna da matriz
	for (int i = idThread; i <= n; i += gridDim.x * blockDim.x) {
		d_dist[(m + 1) * (i + 1)] = i + 1;
	}
}

// Calcula a distância de edição das antidiagonais
__global__
void distancia_edicao_adiag(int antiDiag, int n, int m, const char *d_s, const char *d_r, int *d_dist) {
	int iBloco = blockDim.x * blockIdx.x;
	int jBloco = blockDim.x * (antiDiag - blockIdx.x);
	int nDiagSub = min(blockDim.x, n - iBloco) + min(blockDim.x, m - jBloco);

	if (antiDiag == 0) {
		prepara_matriz(n, m, d_dist);
		__syncthreads();
	}

	// Para cada antidiagonal da submatriz
	for (int antiDiagSub = 2; antiDiagSub <= nDiagSub; antiDiagSub++) {
		// Calcula índices i e j da célula (linha e coluna)
		int i = iBloco + threadIdx.x + 1;
		int j = jBloco + antiDiagSub - threadIdx.x - 1;

		// Se é uma célula válida
		if (i <= n && j <= m && 0 <= jBloco && jBloco + 1 <= j && j <= jBloco + 1 + blockDim.x) {
			int a = d_dist[(m+1) *   i   + j-1] + 1;
			int b = d_dist[(m+1) * (i-1) +  j ] + 1;
			int c = d_dist[(m+1) * (i-1) + j-1] + (d_s[i] != d_r[j]);
			d_dist[(m+1) * i + j] = min(a, b, c);
		}
		__syncthreads();
	}
}

int main(int argc, const char *argv[]) {
	int n, m;     // Tamanho das sequências s e r
	char *h_s;    // Sequência s de entrada (com tamanho n+1)
	char *h_r;    // Sequência r de entrada (com tamanho m+1)
	int *d_dist;  // Matriz de distâncias com tamanho (n+1) * (m+1)
	char *d_s, *d_r;  // Cópias das sequências no device

	if (argc != 2) {
		printf("O programa foi executado com argumentos incorretos.\n");
		printf("Uso: ./dist_seq <nome arquivo entrada>\n");
		return EXIT_FAILURE;
	}

	// Lê sequências do arquivo de entrada
	ler_entrada(argv[1], &n, &m, &h_s, &h_r);

	// Aloca estruturas no device
	aloca_dev(n, m, &d_s, &d_r, &d_dist);

	int nBlocosS = divteto(n, kTamBloco);  // Total de blocos na sequência S
	int nBlocosR = divteto(m, kTamBloco);  // Total de blocos na sequência R

	float tempo_ms = 0;  // Tempo de execução na CPU em milissegundos
	cudaEvent_t d_ini, d_fim;
	cudaEventCreate(&d_ini);
	cudaEventCreate(&d_fim);
	cudaEventRecord(d_ini, 0);

	cudaMemcpy(d_s, h_s, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, (m + 1) * sizeof(char), cudaMemcpyHostToDevice);

	for (int antiDiag = 0; antiDiag < nBlocosS + nBlocosR - 1; antiDiag++) {
		// Calcula distância de edição entre sequências s e r, por antidiagonais
		distancia_edicao_adiag<<<nBlocosS, kTamBloco>>>(antiDiag, n, m, d_s, d_r, d_dist);
		cudaDeviceSynchronize();
	}

	int dist_total = -1;
	cudaMemcpy(&dist_total, d_dist + (n+1) * (m+1) - 1, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", dist_total);

	cudaEventRecord(d_fim, 0);
	cudaEventSynchronize(d_fim);
	cudaEventElapsedTime(&tempo_ms, d_ini, d_fim);
	cudaEventDestroy(d_ini);
	cudaEventDestroy(d_fim);

	printf("%.2f\n", tempo_ms);

	// Libera vetores s e r e matriz de distâncias
	free(h_s);
	free(h_r);
	cudaFree(d_s);
	cudaFree(d_r);
	cudaFree(d_dist);

	return EXIT_SUCCESS;
}
