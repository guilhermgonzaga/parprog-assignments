// ----------------------------------------------------------------------------
// Roteamento usando algoritmo de Lee
//
// Para compilar: g++ -Wall -pedantic -std=c++11 -fopenmp -o rotseq rotseq.cpp
// Para executar: ./rotseq <nome arquivo entrada> <nome arquivo saída>
// ----------------------------------------------------------------------------

#include <climits>
#include <cstdio>
#include <deque>
#include <omp.h>

// ----------------------------------------------------------------------------
// Tipos

typedef struct	// Posição de uma célula do grid
{
	int i, j;
} t_celula;

typedef std::deque<t_celula> queue_t;

// ----------------------------------------------------------------------------
// Variáveis globais

int n_linhas, n_colunas;	// No. de linhas e colunas do grid
int **dist;			// Matriz com distância da origem até cada célula do grid

t_celula origem, destino;

// ----------------------------------------------------------------------------
// Funções

int inicializa(const char *nome_arq_entrada)
{
	int n_obstaculos,		// Número de obstáculos do grid
	    n_linhas_obst,
	    n_colunas_obst;
	t_celula obstaculo;

	FILE *arq_entrada = fopen(nome_arq_entrada, "rt");

	if (arq_entrada == NULL)
	{
		printf("\nArquivo texto de entrada não encontrado\n");
		return 1;
	}

	fscanf(arq_entrada, "%d %d", &n_linhas, &n_colunas);
	fscanf(arq_entrada, "%d %d", &origem.i, &origem.j);
	fscanf(arq_entrada, "%d %d", &destino.i, &destino.j);
	fscanf(arq_entrada, "%d", &n_obstaculos);

	// Aloca grid
	dist = new int*[n_linhas];
	for (int i = 0; i < n_linhas; i++)
		dist[i] = new int[n_colunas];
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
	return 0;
}

// ----------------------------------------------------------------------------

void finaliza(const char *nome_arq_saida, const queue_t& caminho, int distancia_min)
{
	FILE *arq_saida = fopen(nome_arq_saida, "wt");

	// Imprime distância mínima no arquivo de saída
	fprintf(arq_saida, "%d\n", distancia_min);

	// Imprime caminho mínimo no arquivo de saída
	for (auto celula : caminho)
		fprintf(arq_saida, "%d %d\n", celula.i, celula.j);

	fclose(arq_saida);

	// Libera grid
	for (int i = 0; i < n_linhas; i++)
		delete[] dist[i];
	delete[] dist;
}

// ----------------------------------------------------------------------------

bool expansao(queue_t& fila)
{
	bool achou = false;
	t_celula vizinho;

	// Insere célula origem na fila de células a serem tratadas
	fila.push_back(origem);

	// Enquanto fila não está vazia e não chegou na célula destino
	while (!fila.empty() && !achou)
	{
		// Remove primeira célula da fila
		t_celula celula = fila.front();
		fila.pop_front();

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
				fila.push_back(vizinho);
			}

			vizinho.i = celula.i + 1; // Vizinho sul
			vizinho.j = celula.j;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				fila.push_back(vizinho);
			}

			vizinho.i = celula.i; // Vizinho oeste
			vizinho.j = celula.j - 1;

			if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				fila.push_back(vizinho);
			}

			vizinho.i = celula.i; // Vizinho leste
			vizinho.j = celula.j + 1;

			if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1;
				fila.push_back(vizinho);
			}
		}
	}

	return achou;
}

// ----------------------------------------------------------------------------

void traceback(queue_t& caminho)
{
	t_celula celula, vizinho;

	// Constrói caminho mínimo, com células do destino até a origem

	// Inicia caminho com célula destino
	caminho.push_front(destino);

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
			caminho.push_front(vizinho);
		else
		{
			vizinho.i = celula.i + 1; // Sul
			vizinho.j = celula.j;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
				caminho.push_front(vizinho);
			else
			{
				vizinho.i = celula.i; // Oeste
				vizinho.j = celula.j - 1;

				if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
					caminho.push_front(vizinho);
				else
				{
					vizinho.i = celula.i; // Leste
					vizinho.j = celula.j + 1;

					if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
						caminho.push_front(vizinho);
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
	int distancia_min = -1;	// Distância do caminho mínimo de origem a destino

	if(argc != 3)
	{
		printf("O programa foi executado com argumentos incorretos.\n"
		       "Uso: ./rot_seq <nome arquivo entrada> <nome arquivo saída>\n");
		return 1;
	}

	queue_t fila;	// Fila de células a serem tratadas
	queue_t caminho;	// Caminho encontrado

	// Lê arquivo de entrada e inicializa estruturas de dados
	if (inicializa(nome_arq_entrada)) {
		return 1;
	}

	// Fase de expansão: calcula distância da origem até demais células do grid
	double tini = omp_get_wtime();
	bool achou = expansao(fila);
	double tfim = omp_get_wtime();
	printf("%s: %g\n", argv[1], tfim - tini);

	if (achou)
	{
		// Obtém distância do caminho mínimo da origem até destino
		distancia_min = dist[destino.i][destino.j];

		// Fase de traceback: obtém caminho mínimo
		traceback(caminho);
	}

	// Finaliza e escreve arquivo de saída
	finaliza(nome_arq_saida, caminho, distancia_min);

	return 0;
}
