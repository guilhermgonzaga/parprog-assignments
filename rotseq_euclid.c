// ----------------------------------------------------------------------------
// Roteamento usando algoritmo de Lee
//
// Para compilar: gcc -Wall -pedantic -fopenmp -o rotseq rotseq.c
// Para executar: ./rotseq <nome arquivo entrada> <nome arquivo saída>
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>

// ----------------------------------------------------------------------------
// Tipos

typedef struct	// Posição de uma célula do grid
{
	int i, j;
} t_celula;

typedef struct no	// Nó da fila de células a serem tratadas e do caminho encontrado
{
	int i, j;
	struct no *prox;
} t_no;

// ----------------------------------------------------------------------------
// Variáveis globais

bool expansao_invertida = false;

int n_linhas, n_colunas;	// No. de linhas e colunas do grid
int distancia_min;	// Distância do caminho mínimo de origem a destino
int **dist;			// Matriz com distância da origem até cada célula do grid

t_celula origem, destino;

t_no *ini_fila, *fim_fila;	// Início e fim da fila de células a serem tratadas
t_no *ini_caminho, *fim_caminho;	// Início e fim do caminho encontrado

// ----------------------------------------------------------------------------
// Funções

int dist_euclid2(t_celula a, t_celula b)
{
	return (a.i - b.i) * (a.i - b.i)
	     + (a.j - b.j) * (a.j - b.j);
}

void escolhe_direcao()
{
	t_celula centro = {n_linhas/2, n_colunas/2};
	int dist_origem = dist_euclid2(origem, centro);
	int dist_destino = dist_euclid2(destino, centro);

	if (dist_destino > dist_origem)
	{
		t_celula temp = origem;
		origem = destino;
		destino = temp;
		expansao_invertida = true;
	}
}

void inicializa(const char *nome_arq_entrada)
{
	FILE *arq_entrada;	// Arquivo texto de entrada
	int n_obstaculos,		// Número de obstáculos do grid
	    n_linhas_obst,
	    n_colunas_obst;

	t_celula obstaculo;

	arq_entrada = fopen(nome_arq_entrada, "rt");

	if (arq_entrada == NULL)
	{
		printf("\nArquivo texto de entrada não encontrado\n");
		exit(1);
	}

	fscanf(arq_entrada, "%d %d", &n_linhas, &n_colunas);
	fscanf(arq_entrada, "%d %d", &origem.i, &origem.j);
	fscanf(arq_entrada, "%d %d", &destino.i, &destino.j);
	fscanf(arq_entrada, "%d", &n_obstaculos);

	escolhe_direcao();

	// Aloca grid
	dist = malloc(n_linhas * sizeof (int*));
	for (int i = 0; i < n_linhas; i++)
		dist[i] = malloc(n_colunas * sizeof (int));
	// Checar se conseguiu alocar

	// Inicializa grid
	for (int i = 0; i < n_linhas; i++)
		for (int j = 0; j < n_colunas; j++)
			dist[i][j] = INT_MAX;

	dist[origem.i][origem.j] = 0; // Distância da origem até ela mesma é 0

	// Lê obstáculos do arquivo de entrada e preenche grid
	for (int k = 0; k < n_obstaculos; k++)
	{
		fscanf(arq_entrada, "%d %d %d %d", &obstaculo.i, &obstaculo.j,
		                                   &n_linhas_obst, &n_colunas_obst);

		for (int i = obstaculo.i; i < obstaculo.i + n_linhas_obst; i++)
			for (int j = obstaculo.j; j < obstaculo.j + n_colunas_obst; j++)
				dist[i][j] = -1;
	}

	fclose(arq_entrada);

	// Inicializa fila vazia
	ini_fila = NULL;
	fim_fila = NULL;

	// Inicializa caminho vazio
	ini_caminho = NULL;
	fim_caminho = NULL;
}

// ----------------------------------------------------------------------------

void finaliza(const char *nome_arq_saida)
{
	FILE *arq_saida;	// Arquivo texto de saída
	t_no *no;

	arq_saida = fopen(nome_arq_saida, "wt");

	// Imprime distância mínima no arquivo de saída
	fprintf(arq_saida, "%d\n", distancia_min);

	// Imprime caminho mínimo no arquivo de saída
	while (ini_caminho != NULL)
	{
		fprintf(arq_saida, "%d %d\n", ini_caminho->i, ini_caminho->j);

		no = ini_caminho;
		ini_caminho = ini_caminho->prox;

		// Libera nó do caminho
		free(no);
	}

	fclose(arq_saida);

	// Libera grid
	for (int i = 0; i < n_linhas; i++)
		free(dist[i]);
	free(dist);
}

// ----------------------------------------------------------------------------
// Insere célula no fim da fila de células a serem tratadas (fila FIFO)

void insere_fila(t_celula celula)
{
	t_no *no = malloc(sizeof(t_no));
	// Checar se conseguiu alocar

	no->i = celula.i;
	no->j = celula.j;
	no->prox = NULL;

	if (ini_fila == NULL)
		ini_fila = no;
	else
		fim_fila->prox = no;

	fim_fila = no;
}

// ----------------------------------------------------------------------------
// Remove célula do início da fila de células a serem tratadas (fila FIFO)

t_celula remove_fila()
{
	t_celula celula;
	t_no *no;

	no = ini_fila;

	celula.i = no->i;
	celula.j = no->j;

	ini_fila = no->prox;

	if (ini_fila == NULL)
		fim_fila = NULL;

	free(no);

	return celula;
}

// ----------------------------------------------------------------------------
// Insere célula no início do caminho

void insere_ini_caminho(t_celula celula)
{
	t_no *no = malloc(sizeof(t_no));
	// Checar se conseguiu alocar

	no->i = celula.i;
	no->j = celula.j;
	no->prox = ini_caminho;

	ini_caminho = no;
}

// ----------------------------------------------------------------------------
// Insere célula no fim do caminho

void insere_fim_caminho(t_celula celula)
{
	t_no *no = malloc(sizeof(t_no));
	// Checar se conseguiu alocar

	no->i = celula.i;
	no->j = celula.j;
	no->prox = NULL;

	if (ini_caminho == NULL)
		ini_caminho = no;
	else
		fim_caminho->prox = no;

	fim_caminho = no;
}

// ----------------------------------------------------------------------------

bool expansao()
{
	bool achou = false;
	t_celula celula, vizinho;

	// Insere célula origem na fila de células a serem tratadas
	insere_fila(origem);

	// Enquanto fila não está vazia e não chegou na célula destino
	while (ini_fila != NULL && !achou)
	{
		// Remove primeira célula da fila
		celula = remove_fila();

		// Checa se chegou ao destino
		if (celula.i == destino.i && celula.j == destino.j)
			achou = true;
		else
		{
			// Para cada um dos 4 possíveis vizinhos da célula (norte, sul, oeste e leste):
			// se célula vizinha existe e ainda não possui valor de distância,
			// calcula distância e insere vizinho na fila de células a serem tratadas

			vizinho.i = celula.i - 1; // Vizinho norte
			vizinho.j = celula.j;

			if ((vizinho.i >= 0) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				insere_fila(vizinho);
			}

			vizinho.i = celula.i + 1; // Vizinho sul
			vizinho.j = celula.j;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				insere_fila(vizinho);
			}

			vizinho.i = celula.i; // Vizinho oeste
			vizinho.j = celula.j - 1;

			if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				insere_fila(vizinho);
			}

			vizinho.i = celula.i; // Vizinho leste
			vizinho.j = celula.j + 1;

			if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				insere_fila(vizinho);
			}
		}
	}

	return achou;
}

// ----------------------------------------------------------------------------

void traceback()
{
	t_celula celula, vizinho;

	void (*insere_caminho)(t_celula) =
		expansao_invertida ? insere_fim_caminho : insere_ini_caminho;

	// Constrói caminho mínimo, com células do destino até a origem

	// Inicia caminho com célula destino
	insere_caminho(destino);

	celula.i = destino.i;
	celula.j = destino.j;

	// Enquanto não chegou na origem
	while (celula.i != origem.i || celula.j != origem.j)
	{
		// Determina se célula anterior no caminho é vizinho norte, sul, oeste ou leste
		// e insere esse vizinho no início do caminho

		vizinho.i = celula.i - 1; // Norte
		vizinho.j = celula.j;

		if ((vizinho.i >= 0) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
			insere_caminho(vizinho);
		else
		{
			vizinho.i = celula.i + 1; // Sul
			vizinho.j = celula.j;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
				insere_caminho(vizinho);
			else
			{
				vizinho.i = celula.i; // Oeste
				vizinho.j = celula.j - 1;

				if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
					insere_caminho(vizinho);
				else
				{
					vizinho.i = celula.i; // Leste
					vizinho.j = celula.j + 1;

					if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
						insere_caminho(vizinho);
				}
			}
		}
		celula.i = vizinho.i;
		celula.j = vizinho.j;
	}
}

// ----------------------------------------------------------------------------
// Programa principal

int main(int argc, const char **argv)
{
	const char *nome_arq_entrada = argv[1], *nome_arq_saida = argv[2];
	bool achou;

	if(argc != 3)
	{
		printf("O programa foi executado com argumentos incorretos.\n");
		printf("Uso: ./rot_seq <nome arquivo entrada> <nome arquivo saída>\n");
		exit(1);
	}

	// Lê arquivo de entrada e inicializa estruturas de dados
	inicializa(nome_arq_entrada);

	// Fase de expansão: calcula distância da origem até demais células do grid
	double tini = omp_get_wtime();
	achou = expansao();
	double tfim = omp_get_wtime();
	printf("%s: %g\n", argv[1], tfim - tini);

	// Se não encontrou caminho de origem até destino
	if (!achou)
		distancia_min = -1;
	else
	{
		// Obtém distância do caminho mínimo da origem até destino
		distancia_min = dist[destino.i][destino.j];

		// Fase de traceback: obtém caminho mínimo
		traceback();
	}

	// Finaliza e escreve arquivo de saída
	finaliza(nome_arq_saida);

	return 0;
}
