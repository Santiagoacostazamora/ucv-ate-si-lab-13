"""Perceptrón multicapa para la compuerta lógica AND usando scikit-learn."""

from sklearn.neural_network import MLPClassifier
import numpy as np


def entrenar_modelo() -> MLPClassifier:
    """Entrena un MLP para resolver la operación lógica AND."""
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    modelo = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    modelo.fit(x, y)
    return modelo


def probar_modelo(modelo: MLPClassifier) -> None:
    """Imprime las predicciones del modelo entrenado."""
    entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predicciones = modelo.predict(entradas)

    print("=== MLPClassifier - Operación AND ===")
    for entrada, prediccion in zip(entradas, predicciones):
        print(f"Entrada: {entrada.tolist()} -> Predicción: {int(prediccion)}")


if __name__ == "__main__":
    modelo_entrenado = entrenar_modelo()
    probar_modelo(modelo_entrenado)
