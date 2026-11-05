"""Pastas do projeto, em um lugar só. Veja o mapa completo no README.md da raiz."""

import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados")
MODELOS = os.path.join(RAIZ, "modelos")
