import os
import re
from datetime import datetime
from collections import Counter

import pandas as pd
import praw
from prawcore.exceptions import NotFound

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# spaCy para extraer sustantivos
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_ENABLED = True
except ImportError:
    SPACY_ENABLED = False
    print("⚠️ spaCy no instalado: se usarán regex para extraer términos.")

# sklearn para clasificación supervisada
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 🔢 Pedir rango de fechas
print("\n📅 Introduce el rango de fechas para buscar nuevos posts")
print(" Usa formato YYYY-MM-DD o YYYY-MM-DD HH:MM:SS (hora opcional)")

def parse_dt(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Formato inválido: {s}")

while True:
    try:
        FECHA_INICIO = parse_dt(input("Start date: ").strip())
        FECHA_FIN    = parse_dt(input("End date:   ").strip())
        break
    except ValueError as e:
        print(e)

# 1) Configuración segura
CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "EU_Tracker by u/anonymous_user")

SUBREDDITS = [
    "worldnews", "AskEurope", "europe", "europeanunion",
    "politics", "worldpolitics", "news", "EuropeNews",
    "europeanparliament", "geopolitics"
]
LIMIT = 50

KEYWORDS_PRIMARIAS   = ["European", "Union", "Europe", "EU"]
KEYWORDS_SECUNDARIAS = ["What", "Think", "How", "See"]
FEEDBACK_PATS        = [
    re.compile(r'\bfeedback1\b', re.IGNORECASE),
    re.compile(r'\bfeedback2\b', re.IGNORECASE),
]

BASE_DIR    = os.getenv("BASE_DIR", os.getcwd())
DATOS_DIR   = os.path.join(BASE_DIR, "datos")
INPUT_FILE  = os.path.join(DATOS_DIR, "reddit_eu_2025_feedback.xlsx")
OUTPUT_FILE = INPUT_FILE

# 2) Cargar histórico
df_old = pd.read_excel(INPUT_FILE)
if "score" in df_old.columns:
    df_old = df_old.drop(columns=["score"])
for col in ("feedback", "num_comments"):
    if col not in df_old.columns:
        df_old[col] = 0
COLUMNS = ["id", "subreddit", "titulo", "url", "fecha", "feedback", "num_comments"]
df_old = df_old.reindex(columns=COLUMNS)

# 3) Conexión a Reddit
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

# 4) Actualizar comentarios
ids = df_old["id"].dropna().astype(str).tolist()
fullnames = [f"t3_{pid}" for pid in ids]
def batcher(iterable, n=100):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]
for batch in batcher(fullnames, 100):
    for submission in reddit.info(fullnames=batch):
        mask = df_old["id"] == submission.id
        df_old.loc[mask, "num_comments"] = submission.num_comments

# 5) Preparar regex
pats_pri = [re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE) for k in KEYWORDS_PRIMARIAS]
pats_sec = [re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE) for k in KEYWORDS_SECUNDARIAS]

# 6) Scrapeo
existing_ids  = set(df_old["id"])
existing_urls = set(df_old["url"])
posts_new     = []

for sub in SUBREDDITS:
    print(f"🔎 Revisando r/{sub}…")
    try:
        for post in reddit.subreddit(sub).new(limit=LIMIT):
            fecha = datetime.fromtimestamp(post.created_utc)
            if not (FECHA_INICIO <= fecha <= FECHA_FIN):
                continue
            pid = post.id
            url = f"https://reddit.com{post.permalink}"
            if pid in existing_ids or url in existing_urls:
                continue
            title = post.title
            if not any(p.search(title) for p in pats_pri):
                continue
            posts_new.append({
                "id":           pid,
                "subreddit":    sub,
                "titulo":       title,
                "url":          url,
                "fecha":        fecha,
                "feedback":     None,
                "num_comments": post.num_comments
            })
    except NotFound:
        print(f"⚠️ r/{sub} no existe — se omite.")
    except Exception as e:
        print(f"⚠️ Error en r/{sub}: {e}")

df_new = pd.DataFrame(posts_new).reindex(columns=COLUMNS)
if df_new.empty:
    print("❗ No hay posts nuevos. END. Thank you")
    exit()

# 7) Clasificación supervisada
vectorizer = TfidfVectorizer(min_df=2)
X_hist = vectorizer.fit_transform(df_old["titulo"].astype(str))
y_hist = df_old["feedback"].astype(int)
clf = LogisticRegression(max_iter=500)
clf.fit(X_hist, y_hist)
X_new = vectorizer.transform(df_new["titulo"].astype(str))
df_new["feedback"] = clf.predict(X_new).astype(int)

# 8) Guardar Excel
df_all = pd.concat([df_old, df_new], ignore_index=True).reindex(columns=COLUMNS)
df_all.to_excel(OUTPUT_FILE, index=False)
print(f"✔️ Guardado en {OUTPUT_FILE} con {len(df_new)} nuevas filas.")

# 9) Topic list
if input("\nTopic list? (Yes/No): ").strip().lower() != "yes":
    print("\nEND. Thank you")
    exit()

exclude = set(map(str.lower, KEYWORDS_PRIMARIAS + KEYWORDS_SECUNDARIAS))
counter = Counter()
for titulo in df_new["titulo"].astype(str):
    if SPACY_ENABLED:
        doc = nlp(titulo)
        for token in doc:
            w = token.text.lower()
            if token.pos_ in {"NOUN", "PROPN"} and len(w) > 3 and w not in exclude:
                counter[w] += 1
    else:
        for w in re.findall(r"\w+", titulo.lower()):
            if len(w) > 3 and w not in exclude:
                counter[w] += 1

top_terms = [t for t,_ in counter.most_common(10)]
print("\n🔤 Top 10 términos (sustantivos):")
for i, term in enumerate(top_terms, 1):
    print(f" {i}. {term} ({counter[term]} apariciones)")

# 10) Comentarios por término
if input("\nPrint comments? (Yes/No): ").strip().lower() != "yes":
    print("\nEND. Thank you")
    exit()

print("\n💬 Total de comentarios por término en ese rango:")
for term in top_terms:
    mask = df_new["titulo"].str.contains(
        rf"\b{re.escape(term)}\b", flags=re.IGNORECASE, regex=True
    )
    total_com = int(df_new.loc[mask, "num_comments"].sum())
    print(f" - {term}: {total_com} comentarios")

print("\nEND. Thank you")
