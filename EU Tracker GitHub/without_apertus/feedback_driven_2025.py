"""
Reddit EU Tracker - Sistema de Monitorización de Opinión Pública sobre la UE
Analiza posts de Reddit relacionados con la Unión Europea usando ML y NLP
"""

import sys
import subprocess
import warnings
import logging
import os
import re
from datetime import datetime
from collections import Counter
import pandas as pd
import praw
from prawcore.exceptions import NotFound
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning, module='praw')
logging.getLogger('praw').setLevel(logging.ERROR)

# Dependencias opcionales
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_ENABLED = True
except ImportError:
    SPACY_ENABLED = False
    print("⚠️ spaCy no instalado. Se usarán regex para análisis de texto.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Config:
    """Configuración centralizada del sistema"""
    
    # Reddit API - Configura tus credenciales en variables de entorno
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT", "EU_Tracker/1.0")
    
    # Subreddits a monitorizar
    SUBREDDITS = [
        "worldnews", "AskEurope", "europe", "europeanunion", 
        "politics", "worldpolitics", "news", "EuropeNews", 
        "europeanparliament", "geopolitics"
    ]
    
    # Parámetros de búsqueda
    LIMIT = 200  # Posts a revisar por subreddit
    KEYWORDS_PRIMARIAS = ["European", "Union", "Europe", "EU"]
    
    # Stopwords personalizadas para análisis semántico
    STOPWORDS_CUSTOM = [
        "european", "union", "europe", "eu", "year", "years", "month", "months",
        "week", "weeks", "day", "days", "time", "september", "october", "november",
        "december", "january", "february", "march", "april", "may", "june", "july",
        "august", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "state", "states", "president", "parliament", "country", "countries",
        "government", "minister", "ministers", "official", "officials", "leader",
        "leaders", "prime", "commission", "commissioner", "council", "member", "members",
        "nations", "national", "meps", "parliamentary", "says", "said", "more", "over",
        "will", "would", "could", "should", "after", "about", "from", "with", "work",
        "working", "works", "worked", "make", "makes", "making", "made", "take", "takes",
        "taking", "took", "give", "gives", "giving", "gave", "call", "calls", "calling",
        "called", "want", "wants", "wanted", "support", "supports", "supporting",
        "supported", "deal", "deals", "dealing", "trade", "trading", "trader", "news",
        "post", "posts", "reddit", "people", "person", "just", "report", "reports",
        "according", "announced", "announcement", "announcements", "statement",
        "statements", "comments", "comment", "new", "first", "last", "next", "latest",
        "recent", "current", "former", "major", "amid", "plans", "plan", "planned",
        "planning", "https", "www", "com", "vote", "votes", "voting", "voted"
    ]
    
    # Whitelist: frases importantes que SÍ queremos (aunque contengan stopwords)
    WHITELIST_PHRASES = [
        "trade war", "trade deal", "trade agreement", "trade talks",
        "climate change", "climate crisis", "climate action",
        "energy crisis", "energy prices", "migration crisis", "migration policy",
        "brexit deal", "brexit talks", "nord stream", "defense spending"
    ]
    
    # Rutas de archivos
    BASE_DIR = os.getenv("DATA_DIR", "./data")
    DATOS_DIR = os.path.join(BASE_DIR, "datos")
    INPUT_FILE = os.path.join(DATOS_DIR, "reddit_eu_data.xlsx")
    
    @classmethod
    def validate_credentials(cls):
        """Valida que las credenciales de Reddit estén configuradas"""
        if not cls.CLIENT_ID or not cls.CLIENT_SECRET:
            raise ValueError(
                "❌ Credenciales de Reddit no configuradas.\n"
                "Por favor, configura las variables de entorno:\n"
                "  - REDDIT_CLIENT_ID\n"
                "  - REDDIT_CLIENT_SECRET\n"
                "  - REDDIT_USER_AGENT (opcional)\n\n"
                "Obtén las credenciales en: https://www.reddit.com/prefs/apps"
            )


class RedditScraper:
    """Scraper de posts de Reddit relacionados con la UE"""
    
    def __init__(self):
        Config.validate_credentials()
        self.reddit = praw.Reddit(
            client_id=Config.CLIENT_ID,
            client_secret=Config.CLIENT_SECRET,
            user_agent=Config.USER_AGENT
        )
        self.pats_pri = [
            re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE) 
            for k in Config.KEYWORDS_PRIMARIAS
        ]
    
    def cargar_historico(self):
        """Carga datos históricos del Excel"""
        if not os.path.exists(Config.INPUT_FILE):
            print(f"⚠️ Archivo histórico no encontrado: {Config.INPUT_FILE}")
            return pd.DataFrame(columns=[
                "id", "subreddit", "titulo", "url", "fecha", "feedback", "num_comments"
            ])
        
        df = pd.read_excel(Config.INPUT_FILE)
        if "score" in df.columns:
            df = df.drop(columns=["score"])
        
        for col in ("feedback", "num_comments"):
            if col not in df.columns:
                df[col] = 0
        
        df["feedback"] = pd.to_numeric(df["feedback"], errors='coerce').fillna(0).astype(int)
        df["num_comments"] = pd.to_numeric(df["num_comments"], errors='coerce').fillna(0).astype(int)
        
        return df.reindex(columns=[
            "id", "subreddit", "titulo", "url", "fecha", "feedback", "num_comments"
        ])
    
    def actualizar_comentarios(self, df):
        """Actualiza el número de comentarios de posts históricos"""
        ids = df["id"].dropna().astype(str).tolist()
        if not ids:
            return df
        
        fullnames = [f"t3_{pid}" for pid in ids]
        
        def batcher(it, n=100):
            for i in range(0, len(it), n):
                yield it[i:i+n]
        
        print("🔄 Actualizando comentarios...")
        for batch in batcher(fullnames, 100):
            try:
                for submission in self.reddit.info(fullnames=batch):
                    df.loc[df["id"] == submission.id, "num_comments"] = submission.num_comments
            except Exception as e:
                print(f"⚠️ Error actualizando batch: {e}")
        
        return df
    
    def scrape_posts(self, fecha_inicio, fecha_fin, df_old):
        """Busca posts nuevos en el rango de fechas especificado"""
        existing_ids = set(df_old["id"])
        existing_urls = set(df_old["url"])
        posts = []
        
        for sub in Config.SUBREDDITS:
            print(f"🔎 Revisando r/{sub}…")
            try:
                for post in self.reddit.subreddit(sub).new(limit=Config.LIMIT):
                    fecha = datetime.fromtimestamp(post.created_utc)
                    
                    if not (fecha_inicio <= fecha <= fecha_fin):
                        continue
                    
                    pid = post.id
                    url = f"https://reddit.com{post.permalink}"
                    
                    if pid in existing_ids or url in existing_urls:
                        continue
                    
                    if not any(p.search(post.title) for p in self.pats_pri):
                        continue
                    
                    posts.append({
                        "id": pid,
                        "subreddit": sub,
                        "titulo": post.title,
                        "url": url,
                        "fecha": fecha,
                        "feedback": 0,
                        "num_comments": int(post.num_comments)
                    })
                    
            except NotFound:
                print(f"⚠️ r/{sub} no existe — se omite.")
            except Exception as e:
                print(f"⚠️ Error en r/{sub}: {e}")
        
        df = pd.DataFrame(posts)
        if not df.empty:
            df["num_comments"] = df["num_comments"].astype(int)
            df["feedback"] = df["feedback"].astype(int)
        
        return df


class FeedbackClassifier:
    """Clasificador ML para determinar relevancia de posts"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=2)
        self.clf = LogisticRegression(max_iter=500)
    
    def entrenar(self, df_historico):
        """Entrena el clasificador con datos históricos etiquetados"""
        df_valido = df_historico[df_historico["feedback"].notna()].copy()
        
        print(f"📊 Datos disponibles para entrenamiento: {len(df_valido)}")
        
        if len(df_valido) < 10:
            print("⚠️ Pocos datos históricos para entrenar.")
            print("   Se necesitan al menos 10 posts etiquetados.")
            print("   Se asignará feedback=0 por defecto.")
            return False
        
        X = self.vectorizer.fit_transform(df_valido["titulo"].astype(str))
        y = df_valido["feedback"].astype(int)
        self.clf.fit(X, y)
        print(f"✅ Modelo entrenado con {len(df_valido)} ejemplos")
        return True
    
    def predecir(self, df_nuevos):
        """Predice feedback para posts nuevos"""
        if df_nuevos.empty:
            return df_nuevos
        
        X = self.vectorizer.transform(df_nuevos["titulo"].astype(str))
        df_nuevos["feedback"] = self.clf.predict(X).astype(int)
        return df_nuevos


class TopicAnalyzer:
    """Analizador de tópicos con soporte para TF-IDF semántico"""
    
    def __init__(self):
        self.exclude = set(map(str.lower, Config.STOPWORDS_CUSTOM))
    
    def extraer_terminos(self, df, metodo="tfidf"):
        """
        Extrae términos relevantes usando el método especificado
        
        Args:
            df: DataFrame con posts
            metodo: "frecuencia" o "tfidf"
        """
        if metodo == "tfidf":
            return self._extraer_tfidf(df)
        return self._extraer_frecuencia(df)
    
    def _extraer_frecuencia(self, df):
        """Extracción simple por frecuencia de aparición"""
        counter = Counter()
        
        for titulo in df["titulo"].astype(str):
            if SPACY_ENABLED:
                doc = nlp(titulo)
                for token in doc:
                    w = token.text.lower()
                    if token.pos_ in {"NOUN", "PROPN"} and len(w) > 3 and w not in self.exclude:
                        counter[w] += 1
            else:
                for w in re.findall(r"\w+", titulo.lower()):
                    if len(w) > 3 and w not in self.exclude:
                        counter[w] += 1
        
        return counter
    
    def _extraer_tfidf(self, df):
        """Extracción semántica con TF-IDF para detectar términos distintivos"""
        titulos = df["titulo"].astype(str).tolist()
        
        vectorizer = TfidfVectorizer(
            max_features=150,
            min_df=2,
            max_df=0.7,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(titulos)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            terms = list(zip(feature_names, scores))
            
            filtered = []
            whitelist_lower = [p.lower() for p in Config.WHITELIST_PHRASES]
            
            for term, score in terms:
                term_lower = term.lower()
                
                # Priorizar frases en whitelist
                if any(w in term_lower for w in whitelist_lower):
                    filtered.append((term, score * 1.5))
                    continue
                
                # Filtrar stopwords
                words = term_lower.split()
                if len(words) == 1:
                    if term_lower in self.exclude:
                        continue
                else:
                    non_stop = [w for w in words if w not in self.exclude]
                    if len(non_stop) == 0:
                        continue
                
                if len(term_lower) < 4:
                    continue
                
                filtered.append((term, score))
            
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            counter = Counter()
            for term, score in filtered[:15]:
                counter[term] = int(score * 100)
            
            return counter
        
        except Exception as e:
            print(f"⚠️ Error en TF-IDF: {e}. Usando frecuencia simple.")
            return self._extraer_frecuencia(df)
    
    def contar_comentarios_por_termino(self, df, terminos):
        """Cuenta posts y comentarios totales por término"""
        stats = {}
        for term in terminos:
            mask = df["titulo"].str.contains(
                rf"\b{re.escape(term)}\b", 
                flags=re.IGNORECASE, 
                regex=True, 
                na=False
            )
            stats[term] = {
                "posts": int(mask.sum()),
                "comentarios": int(df.loc[mask, "num_comments"].sum())
            }
        return stats


class Visualizer:
    """Generador de visualizaciones y gráficos"""
    
    @staticmethod
    def configurar_estilo():
        """Configura el estilo visual de los gráficos"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    @staticmethod
    def grafico_posts_por_subreddit(df):
        """Gráfico de barras: posts por subreddit"""
        plt.figure(figsize=(12, 6))
        conteo = df['subreddit'].value_counts()
        sns.barplot(x=conteo.values, y=conteo.index, hue=conteo.index, 
                    palette='viridis', legend=False)
        plt.title('Posts por Subreddit', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Posts')
        plt.ylabel('Subreddit')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.DATOS_DIR, 'posts_subreddit.png'), dpi=300)
        plt.show()
    
    @staticmethod
    def grafico_evolucion_temporal(df):
        """Gráfico de línea: evolución temporal de posts"""
        plt.figure(figsize=(14, 6))
        df['fecha'] = pd.to_datetime(df['fecha'])
        df_fecha = df.set_index('fecha').resample('D').size()
        
        plt.plot(df_fecha.index, df_fecha.values, marker='o', linewidth=2)
        plt.title('Evolución Temporal de Posts sobre la UE', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Número de Posts')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.DATOS_DIR, 'evolucion.png'), dpi=300)
        plt.show()
    
    @staticmethod
    def grafico_comentarios_vs_posts(df):
        """Gráfico de barras: top 15 posts por comentarios"""
        plt.figure(figsize=(12, 6))
        df_copy = df.copy()
        df_copy['num_comments'] = pd.to_numeric(
            df_copy['num_comments'], errors='coerce'
        ).fillna(0).astype(int)
        df_sorted = df_copy.nlargest(15, 'num_comments')
        
        sns.barplot(data=df_sorted, y='titulo', x='num_comments', 
                    hue='titulo', palette='rocket', legend=False)
        plt.title('Top 15 Posts por Comentarios', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Comentarios')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.DATOS_DIR, 'top_posts.png'), dpi=300)
        plt.show()
    
    @staticmethod
    def wordcloud_titulos(df):
        """Word cloud de títulos de posts"""
        texto = ' '.join(df['titulo'].astype(str))
        stopwords = set(WordCloud().stopwords)
        stopwords.update(map(str.lower, Config.STOPWORDS_CUSTOM))
        
        wordcloud = WordCloud(
            width=1600, height=800, 
            background_color='white',
            colormap='viridis', 
            max_words=100, 
            stopwords=stopwords
        ).generate(texto)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud de Títulos', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.DATOS_DIR, 'wordcloud.png'), dpi=300)
        plt.show()
    
    @staticmethod
    def grafico_terminos_frecuentes(counter):
        """Gráfico de barras: términos más frecuentes"""
        top = counter.most_common(10)
        terminos, frecuencias = zip(*top)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(frecuencias), y=list(terminos), 
                    hue=list(terminos), palette='mako', legend=False)
        plt.title('Top 10 Términos Más Relevantes', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Score TF-IDF')
        plt.ylabel('Término')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.DATOS_DIR, 'terminos.png'), dpi=300)
        plt.show()


class MenuInteractivo:
    """Interfaz de menú interactivo para el usuario"""
    
    @staticmethod
    def menu():
        """Muestra el menú principal"""
        print("\n" + "="*60)
        print("🇪🇺 REDDIT EU TRACKER")
        print("="*60)
        print("\n1. 📊 Analizar datos existentes")
        print("2. 🔍 Buscar posts nuevos + Analizar")
        print("3. 📈 Generar gráficos")
        print("4. 🔄 Actualizar comentarios")
        print("5. ❌ Salir")
        return input("\nOpción (1-5): ").strip()
    
    @staticmethod
    def fechas(requeridas=True):
        """Solicita rango de fechas al usuario"""
        if not requeridas:
            if input("\n¿Filtrar por fechas? (s/n) [n]: ").strip().lower() != 's':
                return None, None
        
        print("\n📅 Introduce el rango de fechas:")
        fecha_inicio = input("Inicio (YYYY-MM-DD) [2025-09-01]: ").strip() or "2025-09-01"
        fecha_fin = input("Fin (YYYY-MM-DD) [2025-09-30]: ").strip() or "2025-09-30"
        return fecha_inicio, fecha_fin
    
    @staticmethod
    def graficos():
        """Pregunta si generar gráficos"""
        return input("\n📊 ¿Generar gráficos? (s/n) [s]: ").strip().lower() != 'n'
    
    @staticmethod
    def metodo():
        """Selección de método de análisis"""
        print("\n🔬 Método de análisis de términos:")
        print("1. Frecuencia simple (más rápido)")
        print("2. TF-IDF semántico (más preciso)")
        return "tfidf" if input("\n(1/2) [2]: ").strip() != "1" else "frecuencia"
    
    @staticmethod
    def exportar(terminos):
        """Pregunta si exportar posts de un término específico"""
        print("\n" + "="*60)
        print("📥 EXPORTACIÓN DE TÉRMINO ESPECÍFICO")
        print("="*60)
        
        if input("\n¿Exportar posts de un término? (s/n): ").strip().lower() != 's':
            return None
        
        print("\n🔤 Términos disponibles:")
        for i, t in enumerate(terminos, 1):
            print(f" {i}. {t}")
        
        termino = input("\n📝 Escribe el término: ").strip().lower()
        terminos_lower = {t.lower(): t for t in terminos}
        return terminos_lower.get(termino)
    
    @staticmethod
    def exportar_posts(df, termino, fecha_inicio=None, fecha_fin=None):
        """Exporta posts que contienen un término específico a Excel"""
        mask = df["titulo"].str.contains(
            rf"\b{re.escape(termino)}\b", 
            flags=re.IGNORECASE, 
            regex=True, 
            na=False
        )
        df_filtrado = df[mask].copy()
        
        if len(df_filtrado) == 0:
            print(f"\n❌ No hay posts con el término '{termino}'")
            return
        
        df_filtrado = df_filtrado.sort_values('num_comments', ascending=False)
        
        termino_safe = termino.replace(" ", "_").replace("/", "-")
        rango = f"_{fecha_inicio}_a_{fecha_fin}" if fecha_inicio and fecha_fin else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        nombre = f"posts_{termino_safe}{rango}_{timestamp}.xlsx"
        ruta = os.path.join(Config.DATOS_DIR, nombre)
        
        os.makedirs(Config.DATOS_DIR, exist_ok=True)
        df_filtrado.to_excel(ruta, index=False)
        
        print(f"\n✅ Exportación completada:")
        print(f"   📄 {nombre}")
        print(f"   📊 {len(df_filtrado)} posts")
        print(f"   💬 {df_filtrado['num_comments'].sum()} comentarios")


def main():
    """Función principal del programa"""
    print("🇪🇺 Reddit EU Tracker")
    print("="*60)
    
    menu = MenuInteractivo()
    opcion = menu.menu()
    
    if opcion == "5":
        print("\n👋 ¡Hasta luego!")
        return
    
    # Inicializar componentes
    scraper = RedditScraper()
    classifier = FeedbackClassifier()
    analyzer = TopicAnalyzer()
    
    # Opción 1: Análisis de datos existentes
    if opcion == "1":
        print("\n📊 MODO: Análisis de datos existentes")
        df = scraper.cargar_historico()
        
        if len(df) == 0:
            print("❌ No hay datos históricos para analizar")
            return
        
        fecha_inicio, fecha_fin = menu.fechas(requeridas=False)
        df_analisis = df.copy()
        
        if fecha_inicio and fecha_fin:
            df_analisis['fecha'] = pd.to_datetime(df_analisis['fecha'])
            fi_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
            ff_dt = datetime.strptime(fecha_fin, "%Y-%m-%d")
            df_analisis = df_analisis[
                (df_analisis['fecha'] >= fi_dt) & 
                (df_analisis['fecha'] <= ff_dt)
            ]
        
        df_relevantes = df_analisis[df_analisis['feedback'] == 1].copy()
        
        if len(df_relevantes) == 0:
            print("❌ No hay posts relevantes (feedback=1) en ese periodo")
            return
        
        print(f"✅ Analizando {len(df_relevantes)} posts relevantes")
        
        metodo = menu.metodo()
        counter = analyzer.extraer_terminos(df_relevantes, metodo=metodo)
        top_terms = [t for t, _ in counter.most_common(10)]
        stats = analyzer.contar_comentarios_por_termino(df_relevantes, top_terms)
        stats_ordenados = sorted(
            stats.items(), 
            key=lambda x: x[1]['comentarios'], 
            reverse=True
        )
        
        print(f"\n🔤 Top 10 términos ({metodo}):")
        for i, term in enumerate(top_terms, 1):
            print(f" {i}. {term} ({counter[term]})")
        
        print("\n💬 Ordenados por comentarios:")
        for i, (term, data) in enumerate(stats_ordenados, 1):
            print(f" {i}. {term.upper()}: {data['comentarios']} comentarios")
        
        if menu.graficos():
            print("\n📊 Generando gráficos...")
            Visualizer.configurar_estilo()
            df_relevantes['num_comments'] = pd.to_numeric(
                df_relevantes['num_comments'], errors='coerce'
            ).fillna(0).astype(int)
            
            Visualizer.grafico_posts_por_subreddit(df_relevantes)
            Visualizer.grafico_evolucion_temporal(df_relevantes)
            Visualizer.grafico_comentarios_vs_posts(df_relevantes)
            Visualizer.wordcloud_titulos(df_relevantes)
            if counter:
                Visualizer.grafico_terminos_frecuentes(counter)
            print("✅ Gráficos guardados")
        
        termino_sel = menu.exportar(top_terms)
        if termino_sel:
            menu.exportar_posts(df_relevantes, termino_sel, fecha_inicio, fecha_fin)
    
    # Opción 2: Scraping + Análisis
    elif opcion == "2":
        print("\n🔍 MODO: Búsqueda de posts nuevos")
        fecha_inicio, fecha_fin = menu.fechas(requeridas=True)
        
        FI = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        FF = datetime.strptime(fecha_fin, "%Y-%m-%d")
        
        df_old = scraper.cargar_historico()
        print(f"📊 Posts históricos: {len(df_old)}")
        
        df_new = scraper.scrape_posts(FI, FF, df_old)
        
        if df_new.empty:
            print("❗ No se encontraron posts nuevos")
            return
        
        print(f"✅ {len(df_new)} posts nuevos encontrados")
        
        if classifier.entrenar(df_old):
            df_new = classifier.predecir(df_new)
        else:
            df_new["feedback"] = 0
        
        os.makedirs(Config.DATOS_DIR, exist_ok=True)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.reindex(columns=[
            "id", "subreddit", "titulo", "url", "fecha", "feedback", "num_comments"
        ])
        df_all.to_excel(Config.INPUT_FILE, index=False)
        
        print(f"\n💾 Datos guardados en: {Config.INPUT_FILE}")
        print(f"   📊 Total: {len(df_all)}")
        print(f"   🆕 Nuevos: {len(df_new)}")
        
        metodo = menu.metodo()
        counter = analyzer.extraer_terminos(df_new, metodo=metodo)
        
        if counter:
            top_terms = [t for t, _ in counter.most_common(10)]
            stats = analyzer.contar_comentarios_por_termino(df_new, top_terms)
            stats_ordenados = sorted(
                stats.items(), 
                key=lambda x: x[1]['comentarios'], 
                reverse=True
            )
            
            print(f"\n🔤 Top 10 términos ({metodo}):")
            for i, term in enumerate(top_terms, 1):
                print(f" {i}. {term} ({counter[term]})")
            
            print("\n💬 Ordenados por comentarios:")
            for i, (term, data) in enumerate(stats_ordenados, 1):
                print(f" {i}. {term.upper()}: {data['comentarios']} comentarios")
        else:
            top_terms = []
        
        if menu.graficos():
            print("\n📊 Generando gráficos...")
            Visualizer.configurar_estilo()
            dg = df
