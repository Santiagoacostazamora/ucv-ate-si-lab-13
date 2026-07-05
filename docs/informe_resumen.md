# Informe resumen - Perceptrón multicapa AND

## Tema
Perceptrón multicapa y algoritmos de aprendizaje aplicados a redes neuronales artificiales.

## Objetivo
Construir un algoritmo de red neuronal artificial multicapa capaz de aprender la compuerta lógica AND.

## Funcionamiento
Una red neuronal multicapa procesa datos mediante una capa de entrada, una o más capas ocultas y una capa de salida. Cada neurona calcula una suma ponderada de sus entradas, aplica una función de activación y envía el resultado a la siguiente capa.

Durante el entrenamiento, la red compara su salida con la salida esperada. Luego ajusta pesos y sesgos usando backpropagation para reducir el error.

## Datos usados

| x1 | x2 | y AND |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

## Conclusión
El perceptrón multicapa entrenado aprende correctamente la compuerta AND, ya que predice 0 para las tres primeras combinaciones y 1 cuando ambas entradas son 1.
