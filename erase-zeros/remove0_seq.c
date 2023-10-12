// ----------------------------------------------------------------------------
// Remove zeros de um vetor
// Para compilar: gcc remove0_seq.c -o remove0_seq -Wall
// Para executar: ./remove0_seq arquivo_entrada arquivo_saida
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// XXX #include <sys/time.h>


void removeZeros(int n, const int *vIn, int *m, int *vOut) {
	int c = 0;

	for (int i = 0; i < n; i++) {
		if (vIn[i] != 0) {
			vOut[c] = vIn[i];
			c++;
		}
	}
	*m = c;
}


int main(int argc, const char *argv[]) {
	int n;      // Número de elementos do vetor de entrada (com zeros)
	int m;      // Número de elementos do vetor de saída (sem zeros)
	int *vIn;   // Vetor de entrada (com n elementos)
	int *vOut;  // Vetor de saída (com até n elementos)

	// -------------------------------------------------------------------------
	// Inicialização

	if (argc != 3) {
		printf("O programa foi executado com argumentos incorretos.\n");
		printf("Uso: ./remove0_seq arquivo_entrada arquivo_saida\n");
		exit(EXIT_FAILURE);
	}

	// Abre arquivo de entrada
	FILE *arqIn = fopen(argv[1], "rt");

	if (arqIn == NULL) {
		printf("\nArquivo texto de entrada não encontrado\n");
		exit(EXIT_FAILURE);
	}

	// Lê tamanho do vetor de entrada
	fscanf(arqIn, "%d", &n);

	// Aloca vetores de entrada e saída
	vIn = (int *) malloc(n * sizeof(int));
	vOut = (int *) malloc(n * sizeof(int));

	if (vIn == NULL || vOut == NULL) {
		printf("\nErro na alocação de estruturas\n");
		exit(EXIT_FAILURE);
	}

	// Lê vetor do arquivo de entrada
	for (int i = 0; i < n; i++) {
		fscanf(arqIn, "%d", &vIn[i]);
	}

	// Fecha arquivo de entrada
	fclose(arqIn);

	// -------------------------------------------------------------------------
	// Corpo principal do programa

	// Mede instante inicial
	clock_t tIni = clock();
	// XXX struct timeval tIni, tFim;
	// XXX gettimeofday(&tIni, 0);

	// Remove zeros do vetor de entrada, produzindo vetor de saída vOut
	removeZeros(n, vIn, &m, vOut);

	// Mede instante final
	clock_t tFim = clock();
	/* TODO delete
	gettimeofday(&tFim, 0);
	long segundos = tFim.tv_sec - tIni.tv_sec;
	long microsegundos = tFim.tv_usec - tIni.tv_usec;
	double tempoMs = (segundos * 1e3) + (microsegundos * 1e-3);
	*/
	printf("Tempo=%.2fms\n", (tFim - tIni) * 1000.0 / CLOCKS_PER_SEC);

	// -------------------------------------------------------------------------
	// Finalização

	// Cria arquivo de saída
	FILE *arqOut = fopen(argv[2], "wt");

	// Escreve tamanho do vetor de saída
	fprintf(arqOut, "%d\n", m);

	// Escreve vetor do arquivo de saída
	for (int i = 0; i < m; i++) {
		fprintf(arqOut, "%d ", vOut[i]);
	}
	fprintf(arqOut, "\n");

	// Fecha arquivo de saída
	fclose(arqOut);

	// Libera vetores de entrada e saída
	free(vIn);
	free(vOut);

	return EXIT_SUCCESS;
}
