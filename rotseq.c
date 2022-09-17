// ----------------------------------------------------------------------------
// Roteamento usando algoritmo de Lee
//
// Para compilar: gcc rotseq.c -o rotseq -Wall
// Para executar: ./rotseq <nome arquivo entrada> <nome arquivo saída>
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

// ----------------------------------------------------------------------------
// Tipos

typedef struct	// Posição de uma célula do grid
{
	int i, j ;
} t_celula ;

typedef struct no	// Nó da fila de células a serem tratadas e do caminho encontrado
{
	int i, j ;
	struct no *prox ;
} t_no;

// ----------------------------------------------------------------------------
// Variáveis globais

int n_linhas,		// No. de linhas e colunas do grid
	 n_colunas,
	 **dist,			// Matriz com distância da origem até cada célula do grid
	 distancia_min ;	// Distância do caminho mínimo de origem a destino

t_celula origem,	// Células origem e destino no grid
			destino ;

t_no *ini_fila,	// Ponteiros para início e fim da fila de células a serem tratadas (fila FIFO)
	  *fim_fila,
	  *ini_caminho ;	// Ponteiro para início do caminho encontrado

// ----------------------------------------------------------------------------
// Funções

void inicializa(char* nome_arq_entrada)
{
	FILE *arq_entrada ;	// Arquivo texto de entrada
	int n_obstaculos,		// Número de obstáculos do grid
		 n_linhas_obst,
		 n_colunas_obst ;

	t_celula obstaculo ;

	arq_entrada = fopen(nome_arq_entrada, "rt") ;

	if (arq_entrada == NULL)
	{
		printf("\nArquivo texto de entrada não encontrado\n") ;
		exit(1) ;
	}

	fscanf(arq_entrada, "%d %d", &n_linhas, &n_colunas) ;
	fscanf(arq_entrada, "%d %d", &origem.i, &origem.j) ;
	fscanf(arq_entrada, "%d %d", &destino.i, &destino.j) ;
	fscanf(arq_entrada, "%d", &n_obstaculos) ;

	// Aloca grid
	dist = malloc(n_linhas * sizeof (int*)) ;
	for (int i = 0; i < n_linhas; i++)
		dist[i] = malloc(n_colunas * sizeof (int)) ;
	// Checar se conseguiu alocar

	// Inicializa grid
	for (int i = 0 ; i < n_linhas ; i++)
		for (int j = 0 ; j < n_colunas ; j++)
			dist[i][j] = INT_MAX ;

	dist[origem.i][origem.j] = 0 ; // Distância da origem até ela mesma é 0

	// Lê obstáculos do arquivo de entrada e preenche grid
	for (int k = 0 ; k < n_obstaculos ; k++)
	{
		fscanf(arq_entrada, "%d %d %d %d", &obstaculo.i, &obstaculo.j, &n_linhas_obst, &n_colunas_obst) ;

		for (int i = obstaculo.i ; i < obstaculo.i + n_linhas_obst ; i++)
			for (int j = obstaculo.j ; j < obstaculo.j + n_colunas_obst ; j++)
				dist[i][j] = -1 ;
	}

	fclose(arq_entrada) ;

	// Inicializa fila vazia
	ini_fila = NULL ;
	fim_fila = NULL ;

	// Inicializa caminho vazio
	ini_caminho = NULL ;
}

// ----------------------------------------------------------------------------

void finaliza(char* nome_arq_saida)
{
	FILE *arq_saida ;	// Arquivo texto de saída
	t_no *no ;

	arq_saida = fopen(nome_arq_saida, "wt") ;

	// Imprime distância mínima no arquivo de saída
	fprintf(arq_saida, "%d\n", distancia_min) ;

	// Imprime caminho mínimo no arquivo de saída
	while (ini_caminho != NULL)
	{
		fprintf(arq_saida, "%d %d\n", ini_caminho->i, ini_caminho->j) ;

		no = ini_caminho ;
		ini_caminho = ini_caminho->prox ;

		// Libera nó do caminho
		free(no) ;
	}

	fclose(arq_saida) ;

	// Libera grid
	for (int i = 0; i < n_linhas; i++)
		free(dist[i]) ;
	free(dist) ;
}

// ----------------------------------------------------------------------------
// Insere célula no fim da fila de células a serem tratadas (fila FIFO)

void insere_fila(t_celula celula)
{
	t_no *no = malloc(sizeof(t_no)) ;
	// Checar se conseguiu alocar

	no->i = celula.i ;
	no->j = celula.j ;
	no->prox = NULL ;

	if (ini_fila == NULL)
		ini_fila = no ;
	else
		fim_fila->prox = no ;

	fim_fila = no ;
}

// ----------------------------------------------------------------------------
// Remove célula do início da fila de células a serem tratadas (fila FIFO)

t_celula remove_fila()
{
	t_celula celula ;
	t_no *no ;

	no = ini_fila ;

	celula.i = no->i ;
	celula.j = no->j ;

	ini_fila = no->prox ;

	if (ini_fila == NULL)
		fim_fila = NULL ;

	free(no) ;

	return celula ;
}

// ----------------------------------------------------------------------------
// Insere célula no inicio do caminho

void insere_caminho(t_celula celula)
{
	t_no *no = malloc(sizeof(t_no)) ;
	// Checar se conseguiu alocar

	no->i = celula.i ;
	no->j = celula.j ;
	no->prox = ini_caminho ;

	ini_caminho = no ;
}

// ----------------------------------------------------------------------------

bool expansao()
{
	bool achou = false ;
	t_celula celula, vizinho ;

	// Insere célula origem na fila de células a serem tratadas
	insere_fila(origem) ;

	// Enquanto fila não está vazia e não chegou na célula destino
	while (ini_fila != NULL && ! achou)
	{
		// Remove 1a. célula da fila
		celula = remove_fila() ;

		// Checa se chegou ao destino
		if (celula.i == destino.i && celula.j == destino.j)
			achou = true ;
		else
		{
			// Para cada um dos 4 possíveis vizinhos da célula (norte, sul, oeste e leste):
			// se célula vizinha existe e ainda não possui valor de distância,
			// calcula distância e insere vizinho na fila de células a serem tratadas

			vizinho.i = celula.i - 1 ; // Vizinho norte
			vizinho.j = celula.j ;

			if ((vizinho.i >= 0) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1 ;
				insere_fila(vizinho) ;
			}

			vizinho.i = celula.i + 1 ; // Vizinho sul
			vizinho.j = celula.j ;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1 ;
				insere_fila(vizinho) ;
			}

			vizinho.i = celula.i ; // Vizinho oeste
			vizinho.j = celula.j - 1 ;

			if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1 ;
				insere_fila(vizinho) ;
			}

			vizinho.i = celula.i ; // Vizinho leste
			vizinho.j = celula.j + 1 ;

			if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == INT_MAX))
			{
				dist[vizinho.i][vizinho.j] = dist[celula.i][celula.j] + 1 ;
				insere_fila(vizinho) ;
			}
		}
	}

	return achou ;
}

// ----------------------------------------------------------------------------

void traceback()
{
	t_celula celula, vizinho ;

	// Constrói caminho mínimo, com células do destino até a origem

	// Inicia caminho com célula destino
	insere_caminho(destino) ;

	celula.i = destino.i ;
	celula.j = destino.j ;

	// Enquanto não chegou na origem
	while (celula.i != origem.i || celula.j != origem.j)
	{
		// Determina se célula anterior no caminho é vizinho norte, sul, oeste ou leste
		// e insere esse vizinho no início do caminho

		vizinho.i = celula.i - 1 ; // Norte
		vizinho.j = celula.j ;

		if ((vizinho.i >= 0) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
			insere_caminho(vizinho) ;
		else
		{
			vizinho.i = celula.i + 1 ; // Sul
			vizinho.j = celula.j ;

			if ((vizinho.i < n_linhas) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
				insere_caminho(vizinho) ;
			else
			{
				vizinho.i = celula.i ; // Oeste
				vizinho.j = celula.j - 1 ;

				if ((vizinho.j >= 0) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
					insere_caminho(vizinho) ;
				else
				{
					vizinho.i = celula.i ; // Leste
					vizinho.j = celula.j + 1 ;

					if ((vizinho.j < n_colunas) && (dist[vizinho.i][vizinho.j] == dist[celula.i][celula.j] - 1))
						insere_caminho(vizinho) ;
				}
			}
		}
		celula.i = vizinho.i ;
		celula.j = vizinho.j ;
	}
}

// ----------------------------------------------------------------------------
// Programa principal

int main(int argc, char** argv)
{
	char nome_arq_entrada[100],
		  nome_arq_saida[100] ;
	bool achou ;

	if(argc != 3)
	{
		printf("O programa foi executado com argumentos incorretos.\n") ;
		printf("Uso: ./rot_seq <nome arquivo entrada> <nome arquivo saída>\n") ;
		exit(1) ;
	}

	// Obtém nome dos arquivos de entrada e saída
	strcpy(nome_arq_entrada, argv[1]) ;
	strcpy(nome_arq_saida, argv[2]) ;

	// Lê arquivo de entrada e inicializa estruturas de dados
	inicializa(nome_arq_entrada) ;

	// Fase de expansão: calcula distância da origem até demais células do grid
	achou = expansao();

	// Se não encontrou caminho de origem até destino
	if (! achou)
		distancia_min = -1 ;
	else
	{
		// Obtém distância do caminho mínimo da origem até destino
		distancia_min = dist[destino.i][destino.j] ;

		// Fase de traceback: obtém caminho mínimo
		traceback();
	}

	// Finaliza e escreve arquivo de saida
	finaliza(nome_arq_saida) ;

	return 0 ;
}
