#!/usr/bin/env python
# coding: utf-8

# # 1. Import Necessary Libraries
# 
# Make sure  Microsoft Visual C++ is installed on your pc
# 
# Extracting text from pdf and converting to csv

# In[66]:


import pdfplumber
import pandas as pd
import os
import re
from collections import Counter
import nltk
from wordsegment import load, segment

# Ensure nltk sentence tokenizer is downloaded
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load English word segmentation model
load()

# Folder containing the PDFs
pdf_folder = r"C:\Users\sadik\OneDrive\Documenten\Howest\semester6\AI_project\studies\studies"
output_csv_path = r"C:\Users\sadik\OneDrive\Documenten\Howest\semester6\AI_project\Project\all_sentences.csv"

# Function to split long merged words into meaningful words
def split_long_words(text):
    if not isinstance(text, str):
        return text  # Return as is if not a string
    
    words = text.split()  # Split text into individual words
    processed_text = []

    for word in words:
        if len(word) > 10:  # Threshold for long words
            segmented_word = " ".join(segment(word))  # Use wordsegment to break into words
            processed_text.append(segmented_word)
        else:
            processed_text.append(word)
    
    return " ".join(processed_text)

# Function to extract and clean sentences from a PDF
def extract_clean_sentences_from_pdf(pdf_path):
    text_data = []
    headers = Counter()
    footers = Counter()
    filename = os.path.basename(pdf_path)  # Extract file name

    with pdfplumber.open(pdf_path) as pdf:
        # Detect common headers/footers
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                if len(lines) > 2:
                    headers[lines[0]] += 1  # First line as potential header
                    footers[lines[-1]] += 1  # Last line as potential footer

        # Identify the most common header/footer
        common_header = headers.most_common(1)[0][0] if headers else ""
        common_footer = footers.most_common(1)[0][0] if footers else ""

        # Extract and clean text
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                lines = text.split("\n")

                # Remove detected headers and footers
                if len(lines) > 2:
                    if lines[0] == common_header:
                        lines.pop(0)  # Remove header
                    if lines[-1] == common_footer:
                        lines.pop(-1)  # Remove footer

                # Join cleaned lines back into text
                cleaned_text = "\n".join(lines)

                # Further cleanup: remove excessive spaces, page numbers, and metadata
                cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)  # Remove extra spaces
                cleaned_text = re.sub(r"Page \d+", "", cleaned_text)  # Remove page numbers
                cleaned_text = re.sub(r"\n+", " ", cleaned_text)  # Remove extra line breaks

                # Tokenize into sentences
                sentences = sent_tokenize(cleaned_text.strip())

                # Save each sentence as a separate row with filename
                for sentence in sentences:
                    cleaned_sentence = split_long_words(sentence)  # Apply word segmentation
                    text_data.append({
                        "filename": filename,
                        "Page": page_num + 1,
                        "sentence": cleaned_sentence
                    })

    return text_data

# Function to process all PDFs in the folder and save to CSV
def process_all_pdfs(pdf_folder, output_csv_path):
    all_text_data = []

    # Loop through all PDF files in the folder
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):  # Only process PDF files
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f" Processing: {pdf_file}")
            text_data = extract_clean_sentences_from_pdf(pdf_path)
            all_text_data.extend(text_data)  # Append extracted text

    # Save to a single CSV file
    df = pd.DataFrame(all_text_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"All PDFs processed! CSV saved at: {output_csv_path}")

# Run the function to process all PDFs
process_all_pdfs(pdf_folder, output_csv_path)

# Load CSV and display first few rows
df = pd.read_csv(output_csv_path)

# Display the cleaned DataFrame
print(" Extracted and Cleaned Sentences DataFrame:")
display(df.head())  # Display the first few rows of cleaned sentences


# # 2.  Load Your Data
# 
# Load the articles from your CSV file using pandas. 

# In[19]:


import pandas as pd

# Load the data
df= pd.read_csv(r'C:\Users\sadik\OneDrive\Documenten\Howest\semester6\AI_project\project\all_sentences.csv')
df.head()


# ### Removing any personal informtion to anonymize data  

# In[20]:


import spacy
import re
import pandas as pd

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Function to remove personal names, dates, and numbers
def remove_sensitive_info(text):
    if not isinstance(text, str):
        return ""  # Handle non-string values

    doc = nlp(text)

    # Remove PERSON names using spaCy NER
    words = [token.text for token in doc if token.ent_type_ != "PERSON"]

    # Remove dates and numbers
    cleaned_text = " ".join(words)
    cleaned_text = re.sub(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', '', cleaned_text)  # Remove dates (YYYY-MM-DD, DD/MM/YYYY)
    cleaned_text = re.sub(r'\b\d+\b', '', cleaned_text)  # Remove standalone numbers

    return cleaned_text.strip()

# Apply function to remove names, personal info, dates, and numbers from df['sentence_clean']
df['sentence_clean'] = df['sentence'].apply(remove_sensitive_info)

# Display cleaned dataframe
display(df.head())


# # 3. Prepare Your Text Data
# We clean up the text
# - Remove the name of city, country, geography for better outcome
# - Remove special characters (only letters)
# - Convert to lower case
# - Remove stop words
# - Remove words of only one or 2 letters ('a', 'I', at,...)
# - Remove very short sentences
# - Remove urls 
# - use stemming
# - do duplicate sentences
# 
# 
# 

# In[21]:


import spacy
import pandas as pd

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

# Function to remove geographic entities (cities, countries, locations)
def remove_geographical_entities(text):
    if not isinstance(text, str):
        return ""  # Handle missing or non-string values
    
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.ent_type_ not in ["GPE", "LOC", "FAC"]]
    
    return " ".join(filtered_tokens)

# Apply function to remove cities, countries, and geography
df['sentence_clean'] = df['sentence'].apply(remove_geographical_entities)

# Display a few cleaned sentences
df.head()


# In[22]:


import re
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import download
from collections import Counter

# Ensure necessary NLTK resources are downloaded
download('wordnet')
download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Retain important stopwords (do NOT remove these)
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
# Remove all stopwords **except** the important ones
stop_words -= important_stopwords

# Minimum word length threshold
minWordSize = 4

# Initialize the WordNetLemmatizer and PorterStemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preprocessing function to clean sentences
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Handle missing or non-string values
    
    # Remove URLs (http, https, www, etc.)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace non-alphabetic characters with a space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Convert text to lowercase and split into words
    words = text.lower().split()

    # Remove stopwords except for retained important ones
    filtered_words = [word for word in words if word not in stop_words]

    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Apply stemming after lemmatization
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    # Remove short words after stopword removal, except retained important words
    final_words = [w for w in stemmed_words if len(w) >= minWordSize or w in important_stopwords]

    # Remove duplicate words within each sentence (preserving order)
    unique_words = list(dict.fromkeys(final_words))

    # Ensure proper spacing between words
    return " ".join(unique_words)

# Apply preprocessing function to clean sentences
df['sentence_clean'] = df['sentence_clean'].apply(preprocess_text)

# Identify and remove duplicate sentences
sentence_counts = Counter(df['sentence_clean'])

# Remove sentences that appear multiple times
df = df[df['sentence_clean'].map(sentence_counts) == 1]

# Display the first elements after processing
df.head()


# In[23]:


df['sentence_clean'].is_unique


# In[24]:


df = df[df['sentence_clean'].str.len() >=10]


# In[25]:


df.info()


# ### To see how data cleaning looks 

# In[ ]:


import os

# Define the filename for the cleaned data
output_filename = "cleaned_data.csv"

# Get the current folder path
current_folder = os.getcwd()

# Full path to save the file
output_path = os.path.join(current_folder, output_filename)

# Save the cleaned DataFrame to CSV
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"✅ Cleaned data saved at: {output_path}")


# ### Read the clean data 

# In[28]:


import pandas as pd

# Load the data
df= pd.read_csv(r'C:\Users\sadik\OneDrive\Documenten\Howest\semester6\AI_project\project\cleaned_data.csv')
df.head()


# # 4. Initialize and Fit BERTopic
# The good thing with BERTopic is that is does most of the work automatically (Meaning, I do not need to bore you to death with details about how it works behind te scenes.)
# 
# We need to do 3 things
# 1. Initialize BERTopic model
# 2. 'Fit' the model -> this  means: run the model, as you would run a simple linear regression
# 3. Look at the topics via 
# 
# To get started, let's just use the default settings.

# In[29]:


unique_filenames_count = df['filename'].nunique()
print(unique_filenames_count)


# In[30]:


from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize BERTopic model
topic_model = BERTopic(calculate_probabilities=True)

# Fit the model with preprocessed text sentences
topics, probabilities = topic_model.fit_transform(df['sentence_clean'])

# View and inspect topics
topic_model.get_topic_info()


# we get here too much topics  132, later we can make sure that the topic are limited to certain number of topics for better analysis and understanding
# 

# In[31]:


# Initialize BERTopic model
topic_model = BERTopic(calculate_probabilities=True, min_topic_size=10, nr_topics=20)

# Fit the model with preprocessed text sentences
topics, probabilities = topic_model.fit_transform(df['sentence_clean'])

# View and inspect topics
topic_model.get_topic_info()



# In[34]:


topic_model.topics_[:20]


# ### Here we reduce the number of topics with the number of pdf files we have uploaded

# In[35]:


topics = int(unique_filenames_count)
topic_model = BERTopic().fit(df['sentence_clean'])
topic_model.reduce_topics(df['sentence_clean'], nr_topics=(topics))


# In[37]:


print(topic_model.topics_)


# ### Here we can search an attribute that is related to certain topics

# In[38]:


similar_topics, similarity = topic_model.find_topics("stress"); similar_topics


# In[39]:


similar_topics, similarity = topic_model.find_topics("happy"); similar_topics


# In[41]:


topic_model.get_topic(16)


# ### topic limited to the pdf count

# In[42]:


topic_model.get_topic(30)


# # 5. Visualize Topics
# We can call .visualize_topics to create a 2D representation of the topics. The  graph is a plotly interactive graph which can be converted to HTML:
# 
# Note: If you get the error 'ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed', go to terminal and type 'pip install --upgrade nbformat  ' 

# In[43]:


# Visualize topics with an interactive plot
topic_model.visualize_topics()


# You can use the slider to select the topic which then lights up red. If you hover over a topic, then general information is given about the topic, including the size of the topic and its corresponding words.
# 
# We can also ask for a representation of the corresponding words for each topic:

# In[44]:


topic_model.visualize_barchart()


# # 6. Visualize Topic Hierarchy¶
# The topics that were created can be hierarchically reduced. In order to understand the potential hierarchical structure of the topics, we can use scipy.cluster.hierarchy to create clusters and visualize how they relate to one another. We can also see what happens to the topic representations when merging topics. 

# In[45]:


hierarchical_topics = topic_model.hierarchical_topics(df['sentence_clean'])
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)


# If you hover over the black circles, you will see the topic representation at that level of the hierarchy. These representations help you understand the effect of merging certain topics. Some might be logical to merge whilst others might not. Moreover, we can now see which sub-topics can be found within certain larger themes.
# 
# You can also print a text-version of the topic representation at the different levels (a bit less pretty, but maybe easier to read.)

# In[46]:


tree = topic_model.get_topic_tree(hierarchical_topics)
print(tree)


# # 7. Visualize documents
# 
# We can visualize the documents (=texts) inside the topics to see if they were assigned correctly or whether they make sense. To do so, we can use the topic_model.visualize_documents() function. This function recalculates the document embeddings and reduces them to 2-dimensional space for easier visualization purposes. 

# In[47]:


df = df.reset_index(drop=True)  # Reset index to avoid KeyError
topic_model.visualize_documents(df['sentence'].tolist())  # Convert Series to list


# When you hover over a point, you can see which text it is. The color tells you to which topic it belongs. While this is very pretty, it might be useful to be able to just open an excel-file or csv, which contains the original text, with the assigned topic, including the topic words:

# In[48]:


import numpy as np
# Add topics and probabilities to the original DataFrame
df["topic_number"] = np.argmax(probabilities, axis=1)

# Also extract the topic names and assign them to the DataFrame
info = topic_model.get_topic_info()
topic_names = info['Representation']

df['topic_name'] = df['topic_number'].map(topic_names)

# Save the updated DataFrame to a CSV

df['topic_name'] = df['topic_number'].map(topic_names)

# Save to a new CSV file
df.to_csv("studies_lobke_with_topics.csv", index=False)


# In[49]:


df.head()


# We can also see the topic distribution per document = the probability that the text belongs to each topic (if a topic is not included in the graph, the probability is 0). Eg, the topic distribution for the sixth document:(!python starts counting at 0, so 6th =5)

# In[60]:


topic_model.visualize_distribution(probabilities[300])


# # 8. Topics per full article
# 
# We extract the number of times a topic is assigned within the full articles.

# In[61]:


import matplotlib.pyplot as plt

# Calculate the count of times each topic is chosen within each article
article_topic_counts = df.groupby('filename')['topic_number'].value_counts().unstack(fill_value=0)

# Rename columns to 'Topic X'
article_topic_counts.columns = [f'Topic {i}' for i in article_topic_counts.columns]

# Display the table
print(article_topic_counts)

# Plot the distribution for each article
article_topic_counts.plot(kind='bar', stacked=True, figsize=(15, 7))
plt.title('Topic Distribution per Article (Count)')
plt.xlabel('Article')
plt.ylabel('Count')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# We could also do the same, but with proportions in stead of counts.

# In[62]:


import matplotlib.pyplot as plt

# Calculate the proportion of times each topic is chosen within each article
article_topic_proportions = df.groupby('filename')['topic_number'].value_counts(normalize=True).unstack(fill_value=0)

# Rename columns to 'Topic X'
article_topic_proportions.columns = [f'Topic {i}' for i in article_topic_proportions.columns]

# Display the table
print(article_topic_proportions)

# Plot the distribution for each article
article_topic_proportions.plot(kind='bar', stacked=True, figsize=(15, 7))
plt.title('Topic Distribution per Article (Proportion)')
plt.xlabel('Article')
plt.ylabel('Proportion')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#  The highest related topic is child_classroom_behavior it makes sense because we are working with data to see how childeren change of what effects the psychology of child after finishing the elementary school and entering to high school.

# In[ ]:


import nbformat
from nbconvert import PythonExporter

# Replace 'YourNotebookName.ipynb' with the actual filename of your notebook.
notebook_filename = "BERTtopic.ipynb"  
output_filename = notebook_filename.replace(".ipynb", ".py")

# Read the notebook file
with open(notebook_filename, encoding="utf-8") as f:
    nb_node = nbformat.read(f, as_version=4)

# Use PythonExporter to convert the notebook to a Python script
python_exporter = PythonExporter()
(script, resources) = python_exporter.from_notebook_node(nb_node)

# Write the converted Python code to a .py file
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(script)

print(f"Notebook has been saved as {output_filename}")

