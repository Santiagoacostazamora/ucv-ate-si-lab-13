"""Ejecutor principal de la práctica 13."""

from mlp_and_sklearn import entrenar_modelo, probar_modelo
from mlp_and_manual import ejecutar_demo


def main() -> None:
    modelo = entrenar_modelo()
    probar_modelo(modelo)
    print()
    ejecutar_demo()


if __name__ == "__main__":
    main()
