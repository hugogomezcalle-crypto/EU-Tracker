# 🇪🇺 Reddit EU Tracker

Sistema automatizado de monitorización y análisis de opinión pública sobre la Unión Europea en Reddit utilizando Machine Learning y Natural Language Processing.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Descripción

Reddit EU Tracker analiza automáticamente conversaciones sobre la Unión Europea en los principales subreddits políticos, identificando temas emergentes, tendencias y patrones de engagement mediante técnicas avanzadas de NLP y Machine Learning.

### ✨ Características Principales

- 🔍 **Scraping Inteligente**: Monitoriza 10 subreddits políticos (200 posts/subreddit)
- 🤖 **Clasificación ML**: Sistema supervisado (TF-IDF + Logistic Regression) para filtrar contenido relevante
- 📊 **Análisis Semántico Avanzado**: 
  - Extracción de términos con TF-IDF
  - Detección de n-gramas (bigramas y trigramas)
  - Filtrado inteligente con stopwords contextuales y whitelist semántica
- 📈 **Visualizaciones**: Gráficos profesionales (evolución temporal, word clouds, rankings)
- 💾 **Exportación Automatizada**: Genera reportes en Excel por término específico
- 🎯 **Sistema de Feedback**: Etiquetado de relevancia para entrenamiento continuo

## 🚀 Demo

### Análisis de Tópicos
```
🔤 Top 10 términos (tfidf):
 1. migration crisis (score: 352)
 2. trade war china (score: 287)
 3. climate action (score: 245)
 4. sanctions russia (score: 198)
 5. energy prices (score: 176)
```

### Visualizaciones Generadas
- **Evolución Temporal**: Tendencias de posts sobre la UE
- **Word Cloud**: Nube de términos más relevantes
- **Top Posts**: Los 15 posts con más engagement
- **Distribución por Subreddit**: ¿Dónde se habla más de la UE?

## 🛠️ Instalación

### Requisitos
- Python 3.8+
- Credenciales de Reddit API ([Obtenerlas aquí](https://www.reddit.com/prefs/apps))

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/reddit-eu-tracker.git
cd reddit-eu-tracker
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 3: Configurar variables de entorno
Crea un archivo `.env` en la raíz del proyecto:
```bash
REDDIT_CLIENT_ID=tu_client_id
REDDIT_CLIENT_SECRET=tu_client_secret
REDDIT_USER_AGENT=EU_Tracker/1.0
DATA_DIR=./data
```

O exporta las variables directamente:
```bash
export REDDIT_CLIENT_ID="tu_client_id"
export REDDIT_CLIENT_SECRET="tu_client_secret"
export REDDIT_USER_AGENT="EU_Tracker/1.0"
```

### Paso 4: (Opcional) Instalar spaCy para análisis avanzado
```bash
python -m spacy download en_core_web_sm
```

## 📖 Uso

### Ejecución básica
```bash
python reddit_eu_tracker.py
```

### Menú Interactivo
```
🇪🇺 REDDIT EU TRACKER
============================================================

1. 📊 Analizar datos existentes
2. 🔍 Buscar posts nuevos + Analizar
3. 📈 Generar gráficos
4. 🔄 Actualizar comentarios
5. ❌ Salir
```

### Ejemplos de Uso

#### 1. Análisis de datos existentes
Analiza posts ya recopilados, filtra por rango de fechas y genera insights:
```bash
Opción: 1
Filtrar por fechas: s
Inicio: 2025-09-01
Fin: 2025-09-30
Método: 2 (TF-IDF)
```

#### 2. Scraping de posts nuevos
Busca posts nuevos en Reddit y clasifica automáticamente:
```bash
Opción: 2
Inicio: 2025-10-01
Fin: 2025-10-15
```

#### 3. Exportar posts por término
Después del análisis, exporta todos los posts sobre un tema específico:
```bash
¿Exportar posts de un término? s
Términos disponibles:
 1. migration
 2. sanctions
 3. climate
Escribe el término: migration

✅ Exportado: posts_migration_2025-10-01_a_2025-10-15.xlsx
   📊 23 posts
   💬 456 comentarios
```

## 🗂️ Estructura del Proyecto
```
reddit-eu-tracker/
├── reddit_eu_tracker.py    # Script principal
├── requirements.txt         # Dependencias
├── .env.example            # Ejemplo de configuración
├── README.md               # Este archivo
├── data/                   # Directorio de datos (se crea automáticamente)
│   └── datos/
│       ├── reddit_eu_data.xlsx       # Base de datos principal
│       ├── posts_migration_*.xlsx    # Exportaciones
│       ├── posts_subreddit.png       # Gráficos generados
│       ├── evolucion.png
│       ├── wordcloud.png
│       └── terminos.png
└── LICENSE
```

## 🧠 Arquitectura Técnica

### Componentes Principales

#### 1. **RedditScraper**
- Conecta con Reddit API usando PRAW
- Filtra posts por keywords: "European", "Union", "Europe", "EU"
- Deduplicación automática de posts

#### 2. **FeedbackClassifier**
- Clasificador supervisado (TF-IDF + Logistic Regression)
- Entrena con datos históricos etiquetados manualmente
- Predice relevancia de posts nuevos (0 = no relevante, 1 = relevante)

#### 3. **TopicAnalyzer**
- **Método TF-IDF**: Detecta términos distintivos (no solo frecuentes)
- **N-gramas**: Identifica frases completas (ej: "trade war china")
- **Whitelist semántica**: Prioriza términos importantes aunque contengan stopwords
- **60+ stopwords personalizadas**: Filtra palabras genéricas

#### 4. **Visualizer**
- Gráficos profesionales con matplotlib/seaborn
- Exportación en alta resolución (300 DPI)
- Word clouds con filtrado inteligente

### Flujo de Datos
```
┌─────────────────┐
│  Reddit API     │
│  (10 subreddits)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scraper         │
│ Filtra keywords │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classifier ML   │
│ feedback=0/1    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Excel Database  │
│ (acumulativo)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Topic Analyzer  │
│ TF-IDF + NLP    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualizations  │
│ + Exports       │
└─────────────────┘
```

## 📊 Subreddits Monitorizados

- r/worldnews
- r/AskEurope
- r/europe
- r/europeanunion
- r/politics
- r/worldpolitics
- r/news
- r/europeanparliament
- r/geopolitics

## 🎯 Casos de Uso

### Para Analistas Políticos
- Monitorizar opinión pública sobre políticas de la UE
- Detectar temas emergentes antes de que sean tendencia
- Analizar engagement por tema

### Para Investigadores
- Recopilar datos para estudios de opinión pública
- Análisis longitudinal de discurso político
- Identificar patrones en conversaciones sobre la UE

### Para Periodistas
- Descubrir ángulos de noticias basados en conversaciones reales
- Identificar temas con alto engagement
- Exportar evidencia para artículos

## 🔧 Configuración Avanzada

### Modificar Subreddits
Edita la lista en `Config.SUBREDDITS`:
```python
SUBREDDITS = [
    "worldnews", "AskEurope", "europe", "europeanunion",
    # Añade más subreddits aquí
]
```

### Ajustar Stopwords
Añade términos a `Config.STOPWORDS_CUSTOM` para filtrar palabras irrelevantes:
```python
STOPWORDS_CUSTOM = [
    "european", "union", "europe", "eu",
    # Añade más stopwords aquí
]
```

### Personalizar Whitelist
Para priorizar frases específicas aunque contengan stopwords:
```python
WHITELIST_PHRASES = [
    "trade war", "climate change", "migration crisis",
    # Añade más frases importantes aquí
]
```

## 📈 Métricas del Sistema

### Rendimiento
- **Velocidad de scraping**: ~3-5 minutos para 2000 posts
- **Precisión del clasificador**: Depende de datos de entrenamiento (>90% con 1000+ ejemplos etiquetados)
- **Procesamiento NLP**: ~2 segundos para analizar 100 posts con TF-IDF

### Capacidad
- ✅ Procesado 1700+ posts en producción
- ✅ Genera 5 visualizaciones en <10 segundos
- ✅ Soporta hasta 10,000+ posts históricos sin degradación

## 🐛 Solución de Problemas

### Error: "Credenciales no configuradas"
```bash
# Verifica que las variables de entorno estén configuradas
echo $REDDIT_CLIENT_ID
echo $REDDIT_CLIENT_SECRET

# O carga el archivo .env
source .env
```

### Error: "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Sin posts nuevos
- Verifica el rango de fechas (solo busca posts recientes)
- Confirma que los subreddits estén activos
- Revisa las keywords primarias en `Config.KEYWORDS_PRIMARIAS`

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add: AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Ideas para Contribuir
- [ ] Análisis de sentimiento con transformers (BERT/DistilBERT)
- [ ] Dashboard interactivo con Streamlit
- [ ] Base de datos SQL en vez de Excel
- [ ] Notificaciones automáticas (Email/Telegram)
- [ ] Tests unitarios con pytest
- [ ] Deploy en la nube (AWS Lambda/Google Cloud Run)
- [ ] Modo multiidioma (analizar subreddits en otros idiomas)

## 📝 Roadmap

### v1.1 (Próximo)
- [ ] Análisis de sentimiento (positivo/negativo/neutro)
- [ ] API REST para consultas programáticas
- [ ] Dashboard web con Streamlit

### v2.0 (Futuro)
- [ ] Integración con Llama 3.1 para análisis conversacional
- [ ] Soporte para múltiples idiomas
- [ ] Base de datos PostgreSQL
- [ ] Sistema de alertas en tiempo real

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- Email: tu.email@ejemplo.com

## 🙏 Agradecimientos

- [PRAW](https://praw.readthedocs.io/) - Reddit API wrapper
- [scikit-learn](https://scikit-learn.org/) - Machine Learning
- [spaCy](https://spacy.io/) - Natural Language Processing
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) - Visualizaciones

---

⭐ Si este proyecto te resultó útil, ¡dale una estrella en GitHub!

