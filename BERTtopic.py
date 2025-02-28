from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import io
import os
import re
from collections import Counter
import fitz  # PyMuPDF
import nltk
from wordsegment import load, segment
import spacy
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import download
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

# Global variables for topic modeling
topic_model = None
training_texts = None       # Cleaned texts used for training
original_texts = None       # Original texts (if available) for document visualization
training_probabilities = None  # Stored probabilities from training

# -----------------------
# EXTRACTION FUNCTIONS (Using PyMuPDF)
# -----------------------

def is_heading(line: str) -> bool:
    return line.isupper() or line.startswith('CHAPTER')

def is_footnote(line: str) -> bool:
    return re.match(r'^\[\d+\]', line) or re.match(r'^\d+\.', line) or line.startswith('*') or line.startswith('Note') or line.startswith('Table')

def count_words(text: str) -> int:
    return len(text.split())

def contains_doi_or_https(line: str) -> bool:
    return ('doi' in line.lower() or 
            'https' in line.lower() or 
            'http' in line.lower() or 
            'journal' in line.lower() or 
            'university' in line.lower())

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
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    line_text = replace_ligatures(line_text)
                    line_text = fix_common_word_splits(line_text)
                    if is_reference_or_acknowledgements_section(line_text):
                        section_reached = True
                        break
                    if is_heading(line_text) or is_footnote(line_text) or contains_doi_or_https(line_text) or line_text.strip().lower() == filename.lower():
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
# CLEANING FUNCTIONS FOR CSV DATA
# -----------------------

def remove_sensitive_info(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    words = [token.text for token in doc if token.ent_type_ != "PERSON"]
    cleaned_text = " ".join(words)
    cleaned_text = re.sub(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', '', cleaned_text)
    cleaned_text = re.sub(r'\b\d+\b', '', cleaned_text)
    return cleaned_text.strip()

def remove_geographical_entities(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.ent_type_ not in ["GPE", "LOC", "FAC"]]
    return " ".join(filtered_tokens)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.lower().split()
    filtered_words = [word for word in words if not (word in stop_words and word not in important_stopwords)]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    final_words = [w for w in stemmed_words if len(w) >= 4]
    unique_words = list(dict.fromkeys(final_words))
    return " ".join(unique_words)

# -----------------------
# FASTAPI SETUP
# -----------------------
app = FastAPI(
    title="PDF/CSV to BERTopic API",
    description="Upload files until model training. After that, use prediction and visualization endpoints without file uploads."
)

# -----------------------
# API ENDPOINTS
# -----------------------

# Endpoint 1: Combined extraction and cleaning from PDFs
@app.post("/extract_clean")
async def extract_clean(files: List[UploadFile] = File(...), download: bool = Query(True)):
    """
    Upload one or more PDF files or a ZIP file containing PDFs.
    Extracts text using PyMuPDF and then cleans the text using cleaning functions.
    Returns a CSV file with columns: File, Page, text, and text_clean.
    Set download=true to force CSV download; otherwise, returns JSON.
    """
    try:
        all_rows = []
        for file in files:
            if file.filename.lower().endswith(".zip"):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_path = os.path.join(tmpdirname, file.filename)
                    with open(zip_path, "wb") as f:
                        f.write(await file.read())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdirname)
                    for root, _, filenames in os.walk(tmpdirname):
                        for fname in filenames:
                            if fname.lower().endswith(".pdf"):
                                pdf_path = os.path.join(root, fname)
                                extracted = extract_text_from_pdf_fitz(pdf_path)
                                all_rows.extend([{"File": r[0], "Page": r[1], "text": r[2]} for r in extracted])
            elif file.filename.lower().endswith(".pdf"):
                contents = await file.read()
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(contents)
                extracted = extract_text_from_pdf_fitz(temp_path)
                all_rows.extend([{"File": r[0], "Page": r[1], "text": r[2]} for r in extracted])
                os.remove(temp_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        df = pd.DataFrame(all_rows)
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
        raise HTTPException(status_code=500, detail=f"Extract and Clean error: {e}")

# Endpoint 2: Train BERTopic model from a cleaned CSV file
@app.post("/train_model")
async def train_model(file: UploadFile = File(...)):
    """
    Upload a cleaned CSV file (with 'text_clean' or 'text' column) to train a BERTopic model.
    The model, training texts, and original texts (if available) are stored globally.
    """
    global topic_model, training_texts, original_texts, training_probabilities
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if "text_clean" in df.columns:
            text_column = "text_clean"
        elif "text" in df.columns:
            text_column = "text"
        else:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text_clean' or 'text' column.")
        training_texts = df[text_column].astype(str).tolist()
        if "text" in df.columns:
            original_texts = df["text"].astype(str).tolist()
        else:
            original_texts = training_texts
        from bertopic import BERTopic
        # Use nr_topics=20 to get a maximum of 20 topics
        topic_model = BERTopic(calculate_probabilities=True, min_topic_size=5, nr_topics=20)
        topics, training_probabilities = topic_model.fit_transform(training_texts)
        info = topic_model.get_topic_info().to_dict(orient="records")
        return {"detail": "Model trained successfully", "topic_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Training error: {e}")

# Endpoint 3: Predict topic for a single text input
class TextData(BaseModel):
    text: str

@app.post("/predict_topic")
async def predict_topic(data: TextData):
    """
    Provide a JSON payload with a text snippet.
    Returns the predicted topic and its probability using the trained model.
    """
    if topic_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Use /train_model first.")
    try:
        topics, probs = topic_model.transform([data.text])
        pred_topic = topics[0]
        if pred_topic == -1:
            pred_prob = 0.0
        else:
            pred_prob = float(probs[0][pred_topic])
        return {"topic": int(pred_topic), "probability": pred_prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Endpoint 4: Predict topics for an uploaded CSV file
@app.post("/predict_topics_csv")
async def predict_topics_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file (with 'text_clean' or 'text' column) to get topic predictions.
    Returns a CSV file with added 'predicted_topic' and 'probability' columns using the trained model.
    """
    if topic_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Use /train_model first.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if "text_clean" in df.columns:
            text_column = "text_clean"
        elif "text" in df.columns:
            text_column = "text"
        else:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text_clean' or 'text' column.")
        texts = df[text_column].astype(str).tolist()
        topics, probs = topic_model.transform(texts)
        df["predicted_topic"] = topics
        df["probability"] = probs.tolist()
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predicted_topics.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Prediction error: {e}")

# Endpoint 5: Visualize topics (Plotly interactive)
@app.get("/visualize_topics", response_class=HTMLResponse)
async def visualize_topics():
    """
    Returns an interactive Plotly visualization of topics using the trained model.
    """
    if topic_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Use /train_model first.")
    try:
        fig = topic_model.visualize_topics()
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {e}")

# Endpoint 6: Visualize topic barchart (Plotly interactive)
@app.get("/visualize_barchart", response_class=HTMLResponse)
async def visualize_barchart():
    """
    Returns an interactive barchart of topic representations using the trained model.
    """
    if topic_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Use /train_model first.")
    try:
        fig = topic_model.visualize_barchart()
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Barchart Visualization error: {e}")

# Endpoint 7: Visualize topic hierarchy (Plotly interactive)
@app.get("/visualize_hierarchy", response_class=HTMLResponse)
async def visualize_hierarchy():
    """
    Returns an interactive hierarchical visualization of topics using the training texts.
    """
    if topic_model is None or training_texts is None:
        raise HTTPException(status_code=400, detail="Model not trained or training texts not available. Use /train_model first.")
    try:
        hierarchical_topics = topic_model.hierarchical_topics(training_texts)
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hierarchy Visualization error: {e}")

# Endpoint 8: Visualize documents (Plotly interactive)
@app.get("/visualize_documents", response_class=HTMLResponse)
async def visualize_documents():
    """
    Returns an interactive visualization of document embeddings using the original texts (if available) or training texts.
    """
    if topic_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Use /train_model first.")
    try:
        texts_to_use = original_texts if original_texts is not None else training_texts
        fig = topic_model.visualize_documents(texts_to_use)
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Documents Visualization error: {e}")

# Endpoint 9: Visualize overall topic distribution (Plotly interactive)
@app.get("/visualize_distribution", response_class=HTMLResponse)
async def visualize_distribution(doc_index: int = Query(0)):
    """
    Returns an interactive Plotly visualization of the topic distribution.
    If doc_index >= 0, displays the distribution for that document.
    If doc_index is -1, aggregates (averages) probabilities across all documents.
    """
    if topic_model is None or training_texts is None or training_probabilities is None:
        raise HTTPException(status_code=400, detail="Model not trained or training texts/probabilities not available. Use /train_model first.")
    try:
        if doc_index == -1:
            # Aggregate overall probabilities by converting to dense if needed
            all_probs = [prob.toarray() if hasattr(prob, "toarray") else prob for prob in training_probabilities]
            overall_prob = np.mean(np.vstack(all_probs), axis=0)
            fig = topic_model.visualize_distribution(overall_prob)
        else:
            if doc_index < 0 or doc_index >= len(training_texts):
                raise HTTPException(status_code=400, detail="doc_index out of range.")
            prob = training_probabilities[doc_index]
            if hasattr(prob, "toarray"):
                prob = prob.toarray()
            fig = topic_model.visualize_distribution(prob)
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Distribution Visualization error: {e}")

# Endpoint 10: Visualize article topic counts (matplotlib image)
@app.get("/visualize_article_counts")
async def visualize_article_counts():
    """
    Returns a PNG image of a stacked bar chart showing topic counts per article.
    (For demonstration, using dummy filenames if not provided.)
    """
    try:
        if topic_model is None or training_texts is None:
            raise HTTPException(status_code=400, detail="Model not trained or training texts not available.")
        topics, _ = topic_model.transform(training_texts)
        dummy_filenames = ["Article"] * len(training_texts)
        df_sim = pd.DataFrame({"filename": dummy_filenames, "topic_number": topics})
        article_topic_counts = df_sim.groupby('filename')['topic_number'].value_counts().unstack(fill_value=0)
        article_topic_counts.columns = [f'Topic {i}' for i in article_topic_counts.columns]
        plt.figure(figsize=(10, 6))
        article_topic_counts.plot(kind='bar', stacked=True)
        plt.title('Topic Distribution per Article (Count)')
        plt.xlabel('Article')
        plt.ylabel('Count')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Article Counts Visualization error: {e}")

# Endpoint 11: Visualize article topic proportions (matplotlib image)
@app.get("/visualize_article_proportions")
async def visualize_article_proportions():
    """
    Returns a PNG image of a stacked bar chart showing topic proportions per article.
    """
    try:
        if topic_model is None or training_texts is None:
            raise HTTPException(status_code=400, detail="Model not trained or training texts not available.")
        topics, _ = topic_model.transform(training_texts)
        dummy_filenames = ["Article"] * len(training_texts)
        df_sim = pd.DataFrame({"filename": dummy_filenames, "topic_number": topics})
        article_topic_proportions = df_sim.groupby('filename')['topic_number'].value_counts(normalize=True).unstack(fill_value=0)
        article_topic_proportions.columns = [f'Topic {i}' for i in article_topic_proportions.columns]
        plt.figure(figsize=(10, 6))
        article_topic_proportions.plot(kind='bar', stacked=True)
        plt.title('Topic Distribution per Article (Proportion)')
        plt.xlabel('Article')
        plt.ylabel('Proportion')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Article Proportions Visualization error: {e}")

# -----------------------
# RUN THE APPLICATION
# -----------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
