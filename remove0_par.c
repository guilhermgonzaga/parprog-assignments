// ----------------------------------------------------------------------------
// Remove zeros de um vetor
// Estudante: Guilherme Gonzaga de Andrade
//
// Para compilar: mpicc remove0_par.c -o remove0_par -Wall
// Para executar: mpirun -oversubscribe -np 10 remove0_par entrada.txt saida.txt

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define RAIZ 0

// XXX https://github.com/rookiehpc/MPI_monitor

void alocaVetores(int n, int **vIn, int **vOut) {
	*vIn  = (int *) malloc(n * sizeof(int));
	*vOut = (int *) malloc(n * sizeof(int));

	if (*vIn == NULL || *vOut == NULL) {
		printf("\nErro na alocação de estruturas\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
}


void saida(int m, const int *vOut, const char *nomeArq) {
	// Cria arquivo de saída
	FILE *arqOut = fopen(nomeArq, "wt");

	// Escreve tamanho do vetor de saída
	fprintf(arqOut, "%d\n", m);

	// Escreve vetor do arquivo de saída
	for (int i = 0; i < m; i++) {
		fprintf(arqOut, "%d ", vOut[i]);
	}
	fprintf(arqOut, "\n");

	// Fecha arquivo de saída
	fclose(arqOut);
}


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


int main(int argc, char *argv[]) {
	int n = 0;  // Número de elementos do vetor de entrada (com zeros)
	int m = 0;  // Número de elementos do vetor de saída (sem zeros)
	int *vIn = NULL;  // Vetor de entrada (com n elementos)
	int *vOut = NULL;  // Vetor de saída (com até n elementos)
	int rank;
	int nProc;

	// -------------------------------------------------------------------------
	// Inicialização

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);

	if (rank == 0) {
		if (argc != 3) {
			printf("O programa foi executado com argumentos incorretos.\n");
			printf("Uso: ./remove0_seq arquivo_entrada arquivo_saida\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		FILE *arqIn = fopen(argv[1], "rt");

		if (arqIn == NULL) {
			printf("\nArquivo texto de entrada não encontrado\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		// Lê tamanho do vetor de entrada
		fscanf(arqIn, "%d", &n);
		MPI_Bcast(&n, 1, MPI_INT, RAIZ, MPI_COMM_WORLD);

		// Lê vetor do arquivo de entrada
		alocaVetores(n, &vIn, &vOut);
		for (int i = 0; i < n; i++) {
			fscanf(arqIn, "%d", &vIn[i]);
		}

		fclose(arqIn);
	}
	else {
		MPI_Bcast(&n, 1, MPI_INT, RAIZ, MPI_COMM_WORLD);
		alocaVetores(n/nProc, &vIn, &vOut);
	}

	// -------------------------------------------------------------------------
	// Corpo principal do programa

	// Mede instante inicial
	double tIni = MPI_Wtime();

	if (rank == 0) {
		int mParcial[nProc];  // Tamanhos dos vetores de saída distribuídos
		int desloc[nProc];    // Deslocamentos dos dados distribuídos para reunir na raiz

		MPI_Scatter(vIn, n/nProc, MPI_INT, MPI_IN_PLACE, n/nProc, MPI_INT, RAIZ, MPI_COMM_WORLD);

		// Remove zeros do vetor de entrada, produzindo vetor de saída
		removeZeros(n/nProc, vIn, &m, vOut);

		MPI_Gather(&m, 1, MPI_INT, mParcial, 1, MPI_INT, RAIZ, MPI_COMM_WORLD);

		// Soma de prefixos
		desloc[0] = 0;
		for (int i = 1; i < nProc; i++) {
			desloc[i] = desloc[i-1] + mParcial[i-1];
		}

		MPI_Gatherv(MPI_IN_PLACE, n/nProc, MPI_INT, vOut, mParcial, desloc, MPI_INT, RAIZ, MPI_COMM_WORLD/*, &request*/);

		// Atualiza m com o tamanho total enquanto dados estão em escopo
		m = desloc[nProc-1] + mParcial[nProc-1];
	}
	else {
		MPI_Scatter(NULL, n/nProc, MPI_INT, vIn, n/nProc, MPI_INT, RAIZ, MPI_COMM_WORLD);

		// Remove zeros do vetor de entrada, produzindo vetor de saída
		removeZeros(n/nProc, vIn, &m, vOut);

		MPI_Gather(&m, 1, MPI_INT, NULL, 0, MPI_INT, RAIZ, MPI_COMM_WORLD);

		MPI_Gatherv(vOut, m, MPI_INT, NULL, NULL, NULL, MPI_INT, RAIZ, MPI_COMM_WORLD/*, &request*/);
	}

	// MPI_Wait(&request, MPI_STATUS_IGNORE);

	// Mede instante final
	double tFim = MPI_Wtime();
	printf("Tempo=%.2fms\n", 1000.0 * (tFim - tIni));  // TODO o que imprimir?

	// -------------------------------------------------------------------------
	// Finalização

	if (rank == 0) {
		saida(m, vOut, argv[2]);  // Escreve resultado em arquivo
	}

	// Libera vetores de entrada e saída
	free(vIn);
	free(vOut);

	MPI_Finalize();

	return EXIT_SUCCESS;
}
