// ----------------------------------------------------------------------------
// Distância de edição sequencial
//
// Estudante: Guilherme Gonzaga de Andrade
// Estudante:
//
// Para compilar: gcc dist_seq.c -o dist_seq -Wall
// Para executar: ./dist_seq <nome arquivo entrada>
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


int min(int a, int b, int c) {
	if (a < b)
		return (a < c) ? a : c;
	else
		return (b < c) ? b : c;
}


char *aloca_sequencia(int n) {
	char *seq = (char *) malloc((n + 1) * sizeof(char));

	if (seq == NULL) {
		printf("\nErro na alocação de estruturas\n");
		exit(EXIT_FAILURE);
	}

	return seq;
}


int **aloca_matriz(int n, int m) {
	int **mat = (int **) malloc((n + 1) * sizeof(int *));

	if (mat == NULL) {
		printf("\nErro na alocação de estruturas\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i <= n; i++) {
		mat[i] = (int *) malloc((m + 1) * sizeof(int));
		if (mat[i] == NULL) {
			printf("\nErro na alocação de estruturas\n");
			exit(EXIT_FAILURE);
		}
	}
	return mat;
}


void distancia_edicao(int n, int m, const char *s, const char *r, int **d) {
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			int a = d[i][j-1] + 1;
			int b = d[i-1][j] + 1;
			int c = d[i-1][j-1] + (s[i] != r[j]);
			d[i][j] = min(a, b, c);
		}
	}
}


void distancia_edicao_adiagonal(int n, int m, const char *s, const char *r, int **d) {
	int nADiag = n + m - 1;  // Número de anti-diagonais

	// Para cada anti-diagonal
	for (int aD = 2; aD <= nADiag + 1; aD++) {
		// Para cada célula da anti-diagonal aD
		for (int i = n; i >= 0; i--) {
			// Calcula índices i e j da célula (linha e coluna)
			int j = aD - i;

			// Se é uma célula válida
			if (1 <= j && j <= m) {
				int a = d[i][j-1] + 1;
				int b = d[i-1][j] + 1;
				int c = d[i-1][j-1] + (s[i] != r[j]);
				d[i][j] = min(a, b, c);
			}
		}
	}
}


void libera(int n, char *s, char *r, int **d) {
	free(s);
	free(r);
	for (int i = 0; i <= n; i++) {
		free(d[i]);
	}
}


int main(int argc, const char **argv) {
	int n, m;  // Tamanho das sequências s e r

	if (argc != 2) {
		printf("O programa foi executado com argumentos incorretos.\n");
		printf("Uso: ./dist_seq <nome arquivo entrada>\n");
		return EXIT_FAILURE;
	}

	// Abre arquivo de entrada
	FILE *arqEntrada = fopen(argv[1], "rt");

	if (arqEntrada == NULL) {
		printf("\nArquivo texto de entrada não encontrado\n");
		return EXIT_FAILURE;
	}

	// Lê tamanho das sequências s e r
	fscanf(arqEntrada, "%d %d", &n, &m);

	// Aloca vetores s e r
	char *s = aloca_sequencia(n);  // Sequência s de entrada (com tamanho n+1)
	char *r = aloca_sequencia(m);  // Sequência r de entrada (com tamanho m+1)

	// Aloca matriz d
	int **d = aloca_matriz(n, m);  // Matriz de distâncias com tamanho (n+1)*(m+1)

	// Lê sequências do arquivo de entrada
	s[0] = ' ';
	r[0] = ' ';
	fscanf(arqEntrada, "%s", &s[1]);
	fscanf(arqEntrada, "%s", &r[1]);

	// Fecha arquivo de entrada
	fclose(arqEntrada);

	struct timeval h_ini, h_fim;
	gettimeofday(&h_ini, 0);

	// Inicializa matriz de distâncias d
	for (int i = 0; i <= n; i++) {
		d[i][0] = i;
	}
	for (int j = 1; j <= m; j++) {
		d[0][j] = j;
	}

	// Calcula distância de edição entre sequências s e r
	distancia_edicao(n, m, s, r, d);

	// Calcula distância de edição entre sequências s e r, por anti-diagonais
	// distancia_edicao_adiagonal(n, m, s, r, d);

	gettimeofday(&h_fim, 0);
	// Tempo de execução na CPU em milissegundos
	long segundos = h_fim.tv_sec - h_ini.tv_sec;
	long microssegundos = h_fim.tv_usec - h_ini.tv_usec;
	double tempo = (segundos * 1e3) + (microssegundos * 1e-3);

	printf("%d\n", d[n][m]);
	printf("%.2f\n", tempo);

	// Libera vetores s e r e matriz d
	libera(n, s, r, d);

	return EXIT_SUCCESS;
}
