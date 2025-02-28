from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import io
import os
import re
import fitz  # PyMuPDF
import nltk
from nltk import download
from wordsegment import load, segment
import spacy
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
from typing import List
import zipfile
import tempfile
import plotly.express as px
import matplotlib.pyplot as plt
import nbformat
from nbconvert import PythonExporter

# -----------------------
# DOWNLOAD AND LOAD RESOURCES
# -----------------------
nltk.download('punkt')
download('wordnet')
download('stopwords')

load()  # Load wordsegment model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Define the set of important stopwords to retain:
important_stopwords = {
    "not", "never", "nor", "no",
    "can", "could", "should", "would", "may", "might", "must",
    "all", "any", "some", "many", "much", "few", "more", "most", "several", "less", "least",
    "before", "after", "since", "until", "while", "when", "then",
    "than", "as", "like"
}

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# -----------------------
# GLOBAL VARIABLES
# -----------------------
topic_model = None
training_texts = None         # Cleaned texts used for training
original_texts = None         # Original texts (if available) for visualization
training_probabilities = None # Stored probabilities from training

# -----------------------
# HELPER: GET TOPIC LABEL (for last 2 visualizations)
# -----------------------
def get_topic_label(topic_id: int, n_words: int = 3) -> str:
    """
    Return a short label for a topic, e.g. "Topic 0: word1, word2, word3".
    If topic_id == -1, label as 'Outliers'.
    """
    global topic_model
    if topic_id == -1:
        return "Topic -1: Outliers"
    words_and_weights = topic_model.get_topic(topic_id)
    if not words_and_weights:
        return f"Topic {topic_id}"
    top_words = [w for (w, _) in words_and_weights[:n_words]]
    return f"Topic {topic_id}: {', '.join(top_words)}"


# -----------------------
# PDF TEXT EXTRACTION FUNCTIONS (Using PyMuPDF)
# -----------------------
def is_heading(line: str) -> bool:
    return line.isupper() or line.startswith('CHAPTER')

def is_footnote(line: str) -> bool:
    return (
        re.match(r'^\[\d+\]', line) or
        re.match(r'^\d+\.', line) or
        line.startswith('*') or
        line.startswith('Note') or
        line.startswith('Table')
    )

def count_words(text: str) -> int:
    return len(text.split())

def contains_doi_or_https(line: str) -> bool:
    return (
        'doi' in line.lower() or
        'https' in line.lower() or
        'http' in line.lower() or
        'journal' in line.lower() or
        'university' in line.lower()
    )

def is_reference_or_acknowledgements_section(line: str) -> bool:
    markers = ['references', 'bibliography', 'acknowledgements', 'nederlandse', 'method', "methods"]
    return any(marker in line.lower() for marker in markers)

def replace_ligatures(text: str) -> str:
    ligatures = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        'ﬀ': 'ff'
    }
    for lig, repl in ligatures.items():
        text = text.replace(lig, repl)
    return text

def fix_common_word_splits(text: str) -> str:
    fixes = {
        'signi ficant': 'significant',
        'di fferent': 'different',
        'e ffective': 'effective',
        'e ffect': 'effect',
        'chil dren': 'children',
        'e ff ective': 'effective',
        'con fi dence': 'confidence',
    }
    for split_word, correct in fixes.items():
        text = text.replace(split_word, correct)
    text = re.sub(r'\b(\w{3,})\s+(\w{3,})\b', r'\1 \2', text)
    return text

def extract_text_from_pdf_fitz(pdf_path: str):
    """
    Extracts paragraphs from a PDF using PyMuPDF with custom cleaning.
    Returns a list of records: [File, Page, text].
    """
    import fitz
    data = []
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    section_reached = False

    for page_num in range(doc.page_count):
        if section_reached:
            break
        page = doc.load_page(page_num)
        text_dict = page.get_text("dict")

        # Replace semicolons with commas in spans
        for block in text_dict["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span["text"] = span["text"].replace(';', ',')

        for block in text_dict["blocks"]:
            if block["type"] == 0:
                paragraph = []
                prev_x = None
                for line in block["lines"]:
                    line_text = " ".join(span["text"] for span in line["spans"])
                    line_text = replace_ligatures(line_text)
                    line_text = fix_common_word_splits(line_text)

                    if is_reference_or_acknowledgements_section(line_text):
                        section_reached = True
                        break

                    if (
                        is_heading(line_text) or
                        is_footnote(line_text) or
                        contains_doi_or_https(line_text) or
                        line_text.strip().lower() == filename.lower()
                    ):
                        continue

                    first_word_x = line["spans"][0]["bbox"][0]
                    if prev_x is None or abs(first_word_x - prev_x) < 10:
                        paragraph.append(line_text)
                    else:
                        if paragraph and count_words(" ".join(paragraph)) >= 10:
                            data.append([filename, page_num + 1, " ".join(paragraph).strip()])
                        paragraph = [line_text]
                    prev_x = first_word_x

                if paragraph and not section_reached and count_words(" ".join(paragraph)) >= 10:
                    data.append([filename, page_num + 1, " ".join(paragraph).strip()])

    return data

# -----------------------
# CLEANING FUNCTIONS
# -----------------------
def remove_sensitive_info(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    # Remove person names
    words = [token.text for token in doc if token.ent_type_ != "PERSON"]
    cleaned_text = " ".join(words)
    # Remove date-like patterns
    cleaned_text = re.sub(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', '', cleaned_text)
    # Remove standalone numbers
    cleaned_text = re.sub(r'\b\d+\b', '', cleaned_text)
    return cleaned_text.strip()

def remove_geographical_entities(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    # Remove GPE, LOC, FAC
    filtered_tokens = [token.text for token in doc if token.ent_type_ not in ["GPE", "LOC", "FAC"]]
    return " ".join(filtered_tokens)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    # Keep only letters/spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.lower().split()
    # Remove stopwords except important ones
    filtered_words = [
        w for w in words 
        if not (w in stop_words and w not in important_stopwords)
    ]
    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]
    # Stem
    stemmed = [stemmer.stem(w) for w in lemmatized]
    # Keep words of length >= 4
    final_words = [w for w in stemmed if len(w) >= 4]
    # Remove duplicates while preserving order
    unique_words = list(dict.fromkeys(final_words))
    return " ".join(unique_words)

# -----------------------
# FASTAPI INITIALIZATION
# -----------------------
app = FastAPI(
    title="PDF/CSV to BERTopic API",
    description="Upload files until model training. Then use predictions & visualizations."
)

# -----------------------
# 1) EXTRACT + CLEAN
# -----------------------
@app.post("/extract_clean")
async def extract_clean(
    files: List[UploadFile] = File(...),
    download: bool = Query(True)
):
    """
    Upload PDF(s) or a ZIP with PDFs, extracts + cleans them.
    Returns CSV for download or JSON if download=false.
    """
    try:
        all_rows = []
        for file in files:
            # Handle ZIP
            if file.filename.lower().endswith(".zip"):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_path = os.path.join(tmpdirname, file.filename)
                    with open(zip_path, "wb") as f:
                        f.write(await file.read())
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdirname)

                    for root, _, filenames in os.walk(tmpdirname):
                        for fname in filenames:
                            if fname.lower().endswith(".pdf"):
                                pdf_path = os.path.join(root, fname)
                                extracted = extract_text_from_pdf_fitz(pdf_path)
                                all_rows.extend({
                                    "File": r[0], "Page": r[1], "text": r[2]
                                } for r in extracted)

            # Handle single PDF
            elif file.filename.lower().endswith(".pdf"):
                contents = await file.read()
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(contents)
                extracted = extract_text_from_pdf_fitz(temp_path)
                all_rows.extend({
                    "File": r[0], "Page": r[1], "text": r[2]
                } for r in extracted)
                os.remove(temp_path)

            else:
                raise HTTPException(400, f"Unsupported file type: {file.filename}")

        df = pd.DataFrame(all_rows)

        # Clean text
        df["text_clean"] = df["text"].apply(remove_sensitive_info)
        df["text_clean"] = df["text_clean"].apply(remove_geographical_entities)
        df["text_clean"] = df["text_clean"].apply(preprocess_text)

        if download:
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=extracted_cleaned_texts.csv"}
            )
        else:
            return {"extracted_cleaned_texts": df.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(500, f"Extract and Clean error: {e}")

# -----------------------
# 2) TRAIN MODEL (ONLY "text_clean")
# -----------------------
@app.post("/train_model")
async def train_model(file: UploadFile = File(...)):
    """
    Upload a cleaned CSV file (with 'text_clean' column) to train a BERTopic model.
    The model, training texts, and training probabilities are stored globally.
    After training, predictions and visualizations will use the stored model.
    If a 'text' column exists, it is stored as original_texts for visualization; else we use 'text_clean'.
    """
    global topic_model, training_texts, original_texts, training_probabilities
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # We only accept "text_clean"
        if "text_clean" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain a 'text_clean' column."
            )

        # Convert "text_clean" to a list of strings
        training_texts = df["text_clean"].astype(str).tolist()

        # If there's a "text" column, store for visualization
        if "text" in df.columns:
            original_texts = df["text"].astype(str).tolist()
        else:
            original_texts = training_texts

        # Import inside endpoint
        from bertopic import BERTopic

        # Train the model
        topic_model = BERTopic(calculate_probabilities=True, min_topic_size=10, nr_topics=20)
        topics, training_probabilities_var = topic_model.fit_transform(training_texts)
        training_probabilities = training_probabilities_var

        info = topic_model.get_topic_info().to_dict(orient="records")
        return {
            "detail": "Model trained successfully",
            "topic_info": info
        }
    except Exception as e:
        raise HTTPException(500, f"Model Training error: {e}")

# -----------------------
# 3) PREDICT TOPIC (SINGLE TEXT)
# -----------------------
class TextData(BaseModel):
    text: str

@app.post("/predict_topic")
async def predict_topic(data: TextData):
    """
    Predict the topic of a single text snippet.
    """
    if topic_model is None:
        raise HTTPException(400, "Model not trained. Use /train_model first.")
    try:
        topics, probs = topic_model.transform([data.text])
        t = topics[0]
        p = float(probs[0][t]) if t != -1 else 0.0
        return {"topic": int(t), "probability": p}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

# -----------------------
# 4) PREDICT TOPICS (CSV)
# -----------------------
@app.post("/predict_topics_csv")
async def predict_topics_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with 'text_clean' or 'text' column to get topic predictions.
    Returns a CSV with 'predicted_topic' + 'probability' columns.
    """
    if topic_model is None:
        raise HTTPException(400, "Model not trained. Use /train_model first.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # We first look for "text_clean". If not found, fallback to "text"
        if "text_clean" in df.columns:
            text_column = "text_clean"
        elif "text" in df.columns:
            text_column = "text"
        else:
            raise HTTPException(
                400, "CSV must contain either 'text_clean' or 'text' column for prediction."
            )

        texts = df[text_column].astype(str).tolist()
        topics, probs = topic_model.transform(texts)
        df["predicted_topic"] = topics
        df["probability"] = [list(prob_row) for prob_row in probs]

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predicted_topics.csv"}
        )
    except Exception as e:
        raise HTTPException(500, f"CSV Prediction error: {e}")

# -----------------------
# 5) VISUALIZE TOPICS (PLOTLY)
# -----------------------
@app.get("/visualize_topics", response_class=HTMLResponse)
async def visualize_topics():
    """
    Returns an interactive Plotly visualization of topics using the trained model.
    """
    if topic_model is None:
        raise HTTPException(400, "Model not trained. Use /train_model first.")
    try:
        fig = topic_model.visualize_topics()
        return HTMLResponse(fig.to_html(include_plotlyjs="cdn"))
    except Exception as e:
        raise HTTPException(500, f"Visualization error: {e}")

# -----------------------
# 6) VISUALIZE BARCHART (PLOTLY)
# -----------------------
@app.get("/visualize_barchart", response_class=HTMLResponse)
async def visualize_barchart():
    """
    Returns an interactive barchart of topic representations using the trained model.
    """
    if topic_model is None:
        raise HTTPException(400, "Model not trained. Use /train_model first.")
    try:
        fig = topic_model.visualize_barchart()
        return HTMLResponse(fig.to_html(include_plotlyjs="cdn"))
    except Exception as e:
        raise HTTPException(500, f"Barchart Visualization error: {e}")

# -----------------------
# 7) VISUALIZE HIERARCHY (PLOTLY)
# -----------------------
@app.get("/visualize_hierarchy", response_class=HTMLResponse)
async def visualize_hierarchy():
    """
    Returns an interactive hierarchical visualization of topics using the training texts.
    """
    if topic_model is None or training_texts is None:
        raise HTTPException(400, "Model not trained or training texts not available. Use /train_model first.")
    try:
        hierarchical_topics = topic_model.hierarchical_topics(training_texts)
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        return HTMLResponse(fig.to_html(include_plotlyjs="cdn"))
    except Exception as e:
        raise HTTPException(500, f"Hierarchy Visualization error: {e}")

# -----------------------
# 8) VISUALIZE DOCUMENTS (PLOTLY)
# -----------------------
@app.get("/visualize_documents", response_class=HTMLResponse)
async def visualize_documents():
    """
    Returns an interactive 2D visualization of document embeddings using
    the original texts (if available) or training texts.
    """
    if topic_model is None:
        raise HTTPException(400, "Model not trained. Use /train_model first.")
    try:
        docs = original_texts if original_texts else training_texts
        fig = topic_model.visualize_documents(docs)
        return HTMLResponse(fig.to_html(include_plotlyjs="cdn"))
    except Exception as e:
        raise HTTPException(500, f"Documents Visualization error: {e}")

# -----------------------
# 9) VISUALIZE DISTRIBUTION (PLOTLY)
# -----------------------
@app.get("/visualize_distribution", response_class=HTMLResponse)
async def visualize_distribution(doc_index: int = Query(0)):
    """
    Returns an interactive Plotly visualization of the topic distribution.
    doc_index >= 0 => distribution for that doc
    doc_index == -1 => average distribution across all docs
    """
    if topic_model is None or training_texts is None or training_probabilities is None:
        raise HTTPException(400, "Model not trained or training texts/probabilities not available.")
    try:
        if doc_index == -1:
            # Average over all docs
            all_probs = [p.toarray() if hasattr(p, "toarray") else p for p in training_probabilities]
            overall_prob = np.mean(np.vstack(all_probs), axis=0)
            fig = topic_model.visualize_distribution(overall_prob)
        else:
            if doc_index < 0 or doc_index >= len(training_texts):
                raise HTTPException(400, "doc_index out of range.")
            p = training_probabilities[doc_index]
            if hasattr(p, "toarray"):
                p = p.toarray()
            fig = topic_model.visualize_distribution(p)

        return HTMLResponse(fig.to_html(include_plotlyjs="cdn"))
    except Exception as e:
        raise HTTPException(500, f"Distribution Visualization error: {e}")

# -----------------------
# 10) VISUALIZE ARTICLE COUNTS (MATPLOTLIB)
# -----------------------
@app.get("/visualize_article_counts")
async def visualize_article_counts():
    """
    Returns a PNG stacked bar chart showing topic counts per article.
    Using top words in the legend via get_topic_label.
    """
    import matplotlib.pyplot as plt
    import io

    if topic_model is None or training_texts is None:
        raise HTTPException(400, "Model not trained or training texts not available.")

    try:
        topics, _ = topic_model.transform(training_texts)
        dummy_filenames = ["Article"] * len(training_texts)
        df_sim = pd.DataFrame({"File": dummy_filenames, "topic_number": topics})
        article_topic_counts = df_sim.groupby("File")["topic_number"].value_counts().unstack(fill_value=0)

        # Convert numeric IDs to labels
        article_topic_counts.columns = [get_topic_label(c) for c in article_topic_counts.columns]

        plt.figure(figsize=(10, 6))
        article_topic_counts.plot(kind="bar", stacked=True)
        plt.title("Topic Distribution per Article (Count)")
        plt.xlabel("Article")
        plt.ylabel("Count")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Article Counts Visualization error: {e}")

# -----------------------
# 11) VISUALIZE ARTICLE PROPORTIONS (MATPLOTLIB)
# -----------------------
@app.get("/visualize_article_proportions")
async def visualize_article_proportions():
    """
    Returns a PNG stacked bar chart showing topic proportions per article.
    Using top words in the legend via get_topic_label.
    """
    import matplotlib.pyplot as plt
    import io

    if topic_model is None or training_texts is None:
        raise HTTPException(400, "Model not trained or training texts not available.")

    try:
        topics, _ = topic_model.transform(training_texts)
        dummy_filenames = ["Article"] * len(training_texts)
        df_sim = pd.DataFrame({"File": dummy_filenames, "topic_number": topics})
        article_topic_proportions = df_sim.groupby("File")["topic_number"].value_counts(normalize=True).unstack(fill_value=0)

        # Convert numeric IDs to labels
        article_topic_proportions.columns = [get_topic_label(c) for c in article_topic_proportions.columns]

        plt.figure(figsize=(10, 6))
        article_topic_proportions.plot(kind="bar", stacked=True)
        plt.title("Topic Distribution per Article (Proportion)")
        plt.xlabel("Article")
        plt.ylabel("Proportion")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Article Proportions Visualization error: {e}")

# -----------------------
# RUN THE APPLICATION
# -----------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
