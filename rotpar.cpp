// ----------------------------------------------------------------------------
// Roteamento usando algoritmo de Lee
//
// Para compilar: g++ -pedantic -O2 -std=c++11 -fopenmp -o rotpar rotpar.cpp
// Para executar: ./rotpar <nome arquivo entrada> <nome arquivo saída>
// ----------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <deque>
#include <omp.h>

// Valor arbitrário acima de 2
#define INFINITO 127

// ----------------------------------------------------------------------------
// Tipos

typedef struct	// Posição de uma célula do grid
{
	int i, j;
} t_celula;

typedef std::deque<t_celula> queue_t;

// ----------------------------------------------------------------------------
// Variáveis globais

bool expansao_invertida = false;	// Flag para ativar expansão mais excêntrica

int n_linhas, n_colunas;	// No. de linhas e colunas do grid
int8_t **dist;			// Matriz com distância da origem até cada célula do grid

t_celula origem, destino;

// ----------------------------------------------------------------------------
// Funções

// Distância euclidiana ao quadrado
int dist_euclid2(t_celula a, t_celula b)
{
	return (a.i - b.i) * (a.i - b.i)
	     + (a.j - b.j) * (a.j - b.j);
}

// ----------------------------------------------------------------------------

// Expansão mais excêntrica: troca origem e destino durante processamento
// se a distãncia do centro do grid ao destino for maior do que à origem.
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

// ----------------------------------------------------------------------------

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

	escolhe_direcao();

	// Aloca grid
	dist = new int8_t*[n_linhas];
	for (int i = 0; i < n_linhas; i++)
		dist[i] = new int8_t[n_colunas];
	// Checar se conseguiu alocar

	// Inicializa grid
	for (int i = 0; i < n_linhas; i++)
		for (int j = 0; j < n_colunas; j++)
			dist[i][j] = INFINITO;

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

			int8_t prox_dist = (dist[celula.i][celula.j] + 1) % 3;  // Incremento circular

			vizinho.i = celula.i - 1; // Vizinho norte
			vizinho.j = celula.j;

			if ((vizinho.i >= 0) && (dist[vizinho.i][vizinho.j] == INFINITO))
			{
				dist[vizinho.i][vizinho.j] = prox_dist;
				fila.push_back(vizinho);
			}

			vizinho.i = celula.i + 1; // Vizinho sul
			vizinho.j = celula.j;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == INFINITO))
			{
				dist[vizinho.i][vizinho.j] = prox_dist;
				fila.push_back(vizinho);
			}

			vizinho.i = celula.i; // Vizinho oeste
			vizinho.j = celula.j - 1;

			if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == INFINITO))
			{
				dist[vizinho.i][vizinho.j] = prox_dist;
				fila.push_back(vizinho);
			}

			vizinho.i = celula.i; // Vizinho leste
			vizinho.j = celula.j + 1;

			if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == INFINITO))
			{
				dist[vizinho.i][vizinho.j] = prox_dist;
				fila.push_back(vizinho);
			}
		}
	}

	return achou;
}

// ----------------------------------------------------------------------------

int traceback(queue_t& caminho)
{
	t_celula celula, vizinho;
	int dist_total = 0;

	// Ponteiro para função de queue_t seleciona ordem de inserção no caminho
	// com base em flag expansao_invertida
	void (queue_t::*insere)(const t_celula&) = expansao_invertida
		? static_cast<void (queue_t::*)(const t_celula&)>(&queue_t::push_back)
		: static_cast<void (queue_t::*)(const t_celula&)>(&queue_t::push_front);

	// Constrói caminho mínimo, com células do destino até a origem

	// Inicia caminho com célula destino
	(caminho.*insere)(destino);

	celula.i = destino.i;
	celula.j = destino.j;

	// Enquanto não chegou na origem
	while (celula.i != origem.i || celula.j != origem.j)
	{
		dist_total++;

		// Determina se célula anterior no caminho é vizinho norte, sul, oeste ou leste
		// e insere esse vizinho no início do caminho

		int8_t prox_dist = (dist[celula.i][celula.j] + 2) % 3;  // Decremento circular

		vizinho.i = celula.i - 1; // Norte
		vizinho.j = celula.j;

		if ((vizinho.i >= 0) && (dist[vizinho.i][vizinho.j] == prox_dist))
			(caminho.*insere)(vizinho);
		else
		{
			vizinho.i = celula.i + 1; // Sul
			vizinho.j = celula.j;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == prox_dist))
				(caminho.*insere)(vizinho);
			else
			{
				vizinho.i = celula.i; // Oeste
				vizinho.j = celula.j - 1;

				if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == prox_dist))
					(caminho.*insere)(vizinho);
				else
				{
					vizinho.i = celula.i; // Leste
					vizinho.j = celula.j + 1;

					if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == prox_dist))
						(caminho.*insere)(vizinho);
				}
			}
		}
		celula.i = vizinho.i;
		celula.j = vizinho.j;
	}

	return dist_total;
}

// ----------------------------------------------------------------------------
// Programa principal

int main(int argc, const char *argv[])
{
	const char *nome_arq_entrada = argv[1], *nome_arq_saida = argv[2];
	int distancia_min = -1;	// Distância do caminho mínimo de origem a destino

	if(argc != 3)
	{
		printf("O programa foi executado com argumentos incorretos.\n"
		       "Uso: ./rot_seq <nome arquivo entrada> <nome arquivo saída>\n");
		return 1;
	}

	queue_t fila;		// Fila de células a serem tratadas
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
		// Fase de traceback: obtém caminho mínimo
		distancia_min = traceback(caminho);
	}

	// Finaliza e escreve arquivo de saída
	finaliza(nome_arq_saida, caminho, distancia_min);

	return 0;
}
