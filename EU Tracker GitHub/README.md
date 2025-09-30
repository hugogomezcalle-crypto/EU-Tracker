# Reddit EU Feedback Tracker 🇪🇺

Este proyecto permite extraer, clasificar y analizar publicaciones sobre la Unión Europea en Reddit. Utiliza scraping, clasificación supervisada y análisis temático para generar insights a partir de títulos de posts.

## 🧠 ¿Qué hace?

- Scrapea publicaciones recientes de subreddits europeos.
- Clasifica automáticamente el feedback usando TF-IDF + regresión logística.
- Genera listas de términos clave y estadísticas de comentarios.

## ⚙️ Requisitos

- Python 3.8+
- Las siguientes librerías:
  - `pandas`, `praw`, `scikit-learn`, `spacy`, `openpyxl`, `python-dotenv`

Instala todo con:

```bash
pip install -r requirements.txt
