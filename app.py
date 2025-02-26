from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import io
import os
import re
from collections import Counter
import pdfplumber
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
# Load all standard stopwords
stop_words = set(stopwords.words('english'))

# Define the set of important stopwords we want to retain:
important_stopwords = {
    # Negation Words (Critical for meaning)
    "not", "never", "nor", "no",
    # Modality Words (Possibility/Necessity)
    "can", "could", "should", "would", "may", "might", "must",
    # Quantifiers (Define amounts)
    "all", "any", "some", "many", "much", "few", "more", "most", "several", "less", "least",
    # Time References (Useful for context)
    "before", "after", "since", "until", "while", "when", "then",
    # Comparative Words (Relative meaning)
    "than", "as", "like"
}

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Global variables for topic modeling
topic_model = None
training_texts = None  # List of texts used for training

# -----------------------
# UTILITY FUNCTIONS
# -----------------------

# --- PDF Extraction Functions ---
def split_long_words(text: str) -> str:
    if not isinstance(text, str):
        return text
    words = text.split()
    processed_text = []
    for word in words:
        if len(word) > 10:
            segmented_word = " ".join(segment(word))
            processed_text.append(segmented_word)
        else:
            processed_text.append(word)
    return " ".join(processed_text)

def extract_clean_sentences_from_pdf(pdf_path: str):
    text_data = []
    headers = Counter()
    footers = Counter()
    filename = os.path.basename(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        # Detect common headers/footers
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                if len(lines) > 2:
                    headers[lines[0]] += 1
                    footers[lines[-1]] += 1
        common_header = headers.most_common(1)[0][0] if headers else ""
        common_footer = footers.most_common(1)[0][0] if footers else ""
        # Process each page
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                if len(lines) > 2:
                    if lines[0] == common_header:
                        lines.pop(0)
                    if lines[-1] == common_footer:
                        lines.pop(-1)
                cleaned_text = "\n".join(lines)
                cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
                cleaned_text = re.sub(r"Page \d+", "", cleaned_text)
                cleaned_text = re.sub(r"\n+", " ", cleaned_text)
                sentences = nltk.tokenize.sent_tokenize(cleaned_text.strip())
                for sentence in sentences:
                    cleaned_sentence = split_long_words(sentence)
                    text_data.append({
                        "filename": filename,
                        "Page": page_num + 1,
                        "sentence": cleaned_sentence
                    })
    return text_data

# --- Cleaning Functions for CSV Data ---
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
    # Keep words if they are not in stop_words OR if they are in important_stopwords.
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

# 1. Extract sentences from PDFs or a ZIP file containing PDFs
@app.post("/extract_pdfs")
async def extract_pdfs(files: List[UploadFile] = File(...), download: bool = Query(False)):
    """
    Upload one or more PDF files OR a ZIP file containing PDFs.
    Extracts cleaned sentences from each PDF and groups results by source filename.
    If 'download' is True, returns the results as a CSV file; otherwise, returns JSON.
    """
    try:
        all_rows = []
        response = {}
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
                                extracted = extract_clean_sentences_from_pdf(pdf_path)
                                response.setdefault(fname, []).extend(extracted)
                                all_rows.extend(extracted)
            elif file.filename.lower().endswith(".pdf"):
                contents = await file.read()
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(contents)
                extracted = extract_clean_sentences_from_pdf(temp_path)
                response.setdefault(file.filename, []).extend(extracted)
                all_rows.extend(extracted)
                os.remove(temp_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        if download:
            df = pd.DataFrame(all_rows)
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=extracted_sentences.csv"}
            )
        else:
            return {"extracted_sentences": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Extraction error: {e}")

# 2. Clean an uploaded CSV file
@app.post("/clean_csv")
async def clean_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with a 'sentence' column.
    Removes sensitive information, geographic entities, and preprocesses text.
    Returns a CSV file with a new column 'sentence_clean'.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if "sentence" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'sentence' column.")
        df['sentence_clean'] = df['sentence'].apply(remove_sensitive_info)
        df['sentence_clean'] = df['sentence_clean'].apply(remove_geographical_entities)
        df['sentence_clean'] = df['sentence_clean'].apply(preprocess_text)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Cleaning error: {e}")

# 3. Train BERTopic model from an uploaded cleaned CSV file
@app.post("/train_model")
async def train_model(file: UploadFile = File(...)):
    """
    Upload a cleaned CSV file (with 'sentence_clean' or 'text' column) to train a BERTopic model.
    The model and training texts are stored globally.
    After training, predictions and visualizations will use the stored model.
    """
    global topic_model, training_texts
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if "sentence_clean" in df.columns:
            text_column = "sentence_clean"
        elif "text" in df.columns:
            text_column = "text"
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain a 'sentence_clean' or 'text' column."
            )
        training_texts = df[text_column].astype(str).tolist()
        from bertopic import BERTopic
        topic_model = BERTopic(calculate_probabilities=True, min_topic_size=10, nr_topics=20).fit(training_texts)
        info = topic_model.get_topic_info().to_dict(orient="records")
        return {"detail": "Model trained successfully", "topic_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Training error: {e}")

# 4. Predict topic for a single text input
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

# 5. Predict topics for an uploaded CSV file
@app.post("/predict_topics_csv")
async def predict_topics_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file (with 'sentence_clean' or 'text' column) to get topic predictions.
    Returns a CSV file with added 'predicted_topic' and 'probability' columns using the trained model.
    """
    if topic_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Use /train_model first.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if "sentence_clean" in df.columns:
            text_column = "sentence_clean"
        elif "text" in df.columns:
            text_column = "text"
        else:
            raise HTTPException(status_code=400, detail="CSV must contain a 'sentence_clean' or 'text' column.")
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

# 6. Visualize topics (interactive Plotly HTML)
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

# 7. Visualize topic barchart (topic word representation)
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

# 8. Visualize topic hierarchy
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

# 9. Visualize documents (embedding visualization)
@app.get("/visualize_documents", response_class=HTMLResponse)
async def visualize_documents():
    """
    Returns an interactive visualization of document embeddings using the training texts.
    """
    if topic_model is None or training_texts is None:
        raise HTTPException(status_code=400, detail="Model not trained or training texts not available. Use /train_model first.")
    try:
        fig = topic_model.visualize_documents(training_texts)
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Documents Visualization error: {e}")

# 10. Visualize overall topic distribution (aggregated across training texts)
@app.get("/visualize_distribution", response_class=HTMLResponse)
async def visualize_distribution():
    """
    Returns an interactive bar chart of overall topic distribution across all training texts.
    """
    if topic_model is None or training_texts is None:
        raise HTTPException(status_code=400, detail="Model not trained or training texts not available. Use /train_model first.")
    try:
        topics, _ = topic_model.transform(training_texts)
        unique, counts = np.unique(topics, return_counts=True)
        distribution = {int(u): int(c) for u, c in zip(unique, counts)}
        df_dist = pd.DataFrame({"Topic": list(distribution.keys()), "Count": list(distribution.values())})
        fig = px.bar(df_dist, x="Topic", y="Count", title="Overall Topic Distribution")
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overall Distribution Visualization error: {e}")

# 11. Visualize article topic counts (matplotlib image)
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

# 12. Visualize article topic proportions (matplotlib image)
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
