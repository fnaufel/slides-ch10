
library(tidyverse)
library(keras)
library(tfruns)
library(scales)


# Flags -------------------------------------------------------------------

FLAGS <- flags(
  flag_numeric('learning_rate', 0.001),
  flag_numeric('momentum', 0),
  flag_integer('batch_size', 128),
  flag_integer('epochs', 500),
  flag_string('optimizer', 'rmsprop')
)


# Gerar dados normalizados ------------------------------------------------

minimo <- -100
maximo <- 100
dados <- tibble(
  x = rescale(minimo:maximo),
  y = rescale(x^2)
)
  

# Particionar -------------------------------------------------------------

n_dados <- nrow(dados)

prop_treino <- .8
n_treino <- (n_dados * prop_treino) %>% round(0)

dados_treino <- 
  dados[1:n_treino, 'x'] %>% as.matrix()

metas_treino <- 
  dados[1:n_treino, 'y'] %>% as.matrix()

dados_teste <- 
  dados[(n_treino + 1):n_dados, 'x'] %>% as.matrix()

metas_teste <- 
  dados[(n_treino + 1):n_dados, 'y'] %>% as.matrix()


# Criar modelo ------------------------------------------------------------

rede1 <- keras_model_sequential() %>% 
  layer_dense(40, activation = 'relu', input_shape = 1) %>% 
  layer_dense(20, activation = 'relu') %>% 
  layer_dense(1)


# Compilar ----------------------------------------------------------------

if (FLAGS$optimizer == 'rmsprop') {
  opt <- optimizer_rmsprop(
    learning_rate = FLAGS$learning_rate, 
    momentum = FLAGS$momentum
  )
} else if (FLAGS$optimizer == 'adam') {
  opt <- optimizer_adam(
    learning_rate = FLAGS$learning_rate
  )
}

rede1 %>% 
  compile(
    optimizer = opt,
    loss = 'mse',
    metrics = 'mae'
  )


# Treinar -----------------------------------------------------------------

# TODO: Callback para salvar
historico1 <- rede1 %>% fit(
  dados_treino, 
  metas_treino, 
  epochs = FLAGS$epochs,
  batch_size = FLAGS$batch_size,
  validation_split = 0.2,
  verbose = 0
)

