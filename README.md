# Guía Práctica 13 - Perceptrón Multicapa AND

Proyecto desarrollado en Python para construir, entrenar y probar un Perceptrón Multicapa (MLP) que aprende la compuerta lógica AND.

## Contenido

- `src/mlp_and_sklearn.py`: solución con `MLPClassifier` de scikit-learn.
- `src/mlp_and_manual.py`: implementación manual de una red neuronal multicapa con NumPy y backpropagation.
- `src/main.py`: ejecuta ambas soluciones.
- `requirements.txt`: librerías necesarias.
- `docs/informe_resumen.md`: explicación breve del funcionamiento.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Ejecución

```bash
python src/main.py
```

También puedes ejecutar cada archivo por separado:

```bash
python src/mlp_and_sklearn.py
python src/mlp_and_manual.py
```

## Resultado esperado

La salida debe mostrar las predicciones para las cuatro entradas posibles de AND:

| Entrada | Salida AND esperada |
|---|---|
| [0, 0] | 0 |
| [0, 1] | 0 |
| [1, 0] | 0 |
| [1, 1] | 1 |
