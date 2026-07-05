"""Perceptrón multicapa manual para la compuerta lógica AND.

Arquitectura:
- 2 neuronas de entrada
- 2 neuronas en capa oculta
- 1 neurona de salida
- Función de activación sigmoide
- Aprendizaje por descenso del gradiente y backpropagation
"""

import numpy as np


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def derivada_sigmoide(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)


class MLPAndManual:
    def __init__(self, tasa_aprendizaje: float = 0.5, epocas: int = 10000) -> None:
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(size=(2, 2))
        self.b1 = np.zeros((1, 2))
        self.w2 = rng.normal(size=(2, 1))
        self.b2 = np.zeros((1, 1))

    def entrenar(self, x: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self.epocas):
            # Propagación hacia adelante
            z1 = x @ self.w1 + self.b1
            a1 = sigmoide(z1)
            z2 = a1 @ self.w2 + self.b2
            a2 = sigmoide(z2)

            # Error y backpropagation
            error = y - a2
            delta2 = error * derivada_sigmoide(a2)
            delta1 = (delta2 @ self.w2.T) * derivada_sigmoide(a1)

            # Actualización de pesos y sesgos
            self.w2 += self.tasa_aprendizaje * (a1.T @ delta2)
            self.b2 += self.tasa_aprendizaje * np.sum(delta2, axis=0, keepdims=True)
            self.w1 += self.tasa_aprendizaje * (x.T @ delta1)
            self.b1 += self.tasa_aprendizaje * np.sum(delta1, axis=0, keepdims=True)

    def predecir_probabilidad(self, x: np.ndarray) -> np.ndarray:
        a1 = sigmoide(x @ self.w1 + self.b1)
        a2 = sigmoide(a1 @ self.w2 + self.b2)
        return a2

    def predecir(self, x: np.ndarray) -> np.ndarray:
        probabilidades = self.predecir_probabilidad(x)
        return (probabilidades >= 0.5).astype(int)


def ejecutar_demo() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    modelo = MLPAndManual(tasa_aprendizaje=0.5, epocas=10000)
    modelo.entrenar(x, y)

    predicciones = modelo.predecir(x)
    probabilidades = modelo.predecir_probabilidad(x)

    print("=== MLP Manual con NumPy - Operación AND ===")
    for entrada, probabilidad, prediccion in zip(x, probabilidades, predicciones):
        print(
            f"Entrada: {entrada.tolist()} -> "
            f"Probabilidad: {float(probabilidad[0]):.4f} -> "
            f"Predicción: {int(prediccion[0])}"
        )


if __name__ == "__main__":
    ejecutar_demo()
