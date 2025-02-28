#!/usr/bin/env python
# coding: utf-8

# # 1. Import Necessary Libraries
# 
# Make sure  Microsoft Visual C++ is installed on your pc
# 
# Extracting text from pdf and converting to csv

# In[3]:


import fitz  # PyMuPDF
import os
import pandas as pd
import re

# Path to your directory containing the PDFs
doc_dir = r'C:\Users\sadik\OneDrive\Documenten\Howest\semester6\AI_project\studies\studies'

# List to store the blocks of text (as individual records)
data = []

# Function to check if a line is a heading (all uppercase or starts with 'CHAPTER')
def is_heading(line):
    return line.isupper() or line.startswith('CHAPTER')

# Function to check if a line is a footnote (starts with number in brackets, number, or asterisk)
def is_footnote(line):
    return re.match(r'^\[\d+\]', line) or re.match(r'^\d+\.', line) or line.startswith('*') or line.startswith('Note') or line.startswith('Table') 

# Function to count words in a block of text
def count_words(text):
    return len(text.split())

# Function to filter out lines containing DOI, URLs, specific keywords, or phrases
def contains_doi_or_https(line):
    return ('doi' in line.lower() or 
            'https' in line.lower() or 
            'http' in line.lower() or 
            'journal' in line.lower() or 
            'university' in line.lower() or 
            'brookville' in line.lower() or
            'to cite this article' in line.lower() or
            'full terms & conditions' in line.lower() or
            'taylor & francis' in line.lower() or
            'elsevier' in line.lower() or
            'published by' in line.lower() or
            'received' in line.lower() or
            'revised' in line.lower() or
            'author(s)' in line.lower() or
            'source:' in line.lower() or
            'history:' in line.lower() or
            'keywords' in line.lower() or
            'vol.' in line.lower() or 
            'volume' in line.lower() or 
            'downloaded' in line.lower() or    
            'article' in line.lower() or
            'creative commons use' in line.lower() or
            'author' in line.lower() or 
            'copyrighted' in line.lower() or
            'quarterly' in line.lower() or
            'journal' in line.lower() or
            'purtell' in line.lower() or
            'resources:' in line.lower() or
            'publisher' in line.lower() or
            'ying' in line.lower() or
            'cincinnati' in line.lower() or
            'ISSN' in line.lower() or
            'All rights reserved' in line.lower() or
            'authors' in line.lower())

# Function to check if a line is part of the reference or acknowledgements section
def is_reference_or_acknowledgements_section(line):
    reference_markers = ['references', 'bibliography', 'acknowledgements', 'nederlandse', 'method',"methods"]
    return any(marker in line.lower() for marker in reference_markers)

# Function to replace ligatures with their individual characters
def replace_ligatures(text):
    ligatures = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        'ﬀ': 'ff',
        'ﬂ': 'fl',
    }
    for ligature, replacement in ligatures.items():
        text = text.replace(ligature, replacement)
    return text

# Function to fix common word splits
def fix_common_word_splits(text):
    common_fixes = {
        'signi ficant': 'significant',
        'di fferent': 'different',
        'e ffective': 'effective',
        'e ffect': 'effect',
        'chil dren': 'children',
        'e ff ective': 'effective',
        'con fi dence': 'confidence',
    }
    for split_word, correct_word in common_fixes.items():
        text = text.replace(split_word, correct_word)
    
    text = re.sub(r'\b(\w{3,})\s+(\w{3,})\b', r'\1 \2', text)  # Adjust spaces if needed
    return text

# Loop through each file in the directory
for filename in os.listdir(doc_dir):
    if filename.endswith('.pdf'):  # Only process PDF files
        file_path = os.path.join(doc_dir, filename)

        # Extract the title of the PDF (filename without the '.pdf' extension)
        title = os.path.splitext(filename)[0]

        # Open the PDF file using PyMuPDF
        pdf_document = fitz.open(file_path)

        # Flag to indicate if we are in the reference or acknowledgements section for the entire document
        section_reached = False

        # Iterate through each page in the PDF
        for page_num in range(pdf_document.page_count):
            if section_reached:
                break  # Stop processing further pages if the section marker was reached

            page = pdf_document.load_page(page_num)  # Load a page by page number
            text_dict = page.get_text("dict")  # Extract text in dictionary format to preserve layout
            
            # Substitute all semicolons (;) with commas (,)
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # Type 0 is a text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span["text"] = span["text"].replace(';', ',')
            
            # Process each block of text on the page
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # Type 0 is a text block
                    block_text = ""
                    prev_x = None  # To store the previous x-coordinate (indentation level)
                    paragraph = []  # List to store lines that belong to the same paragraph

                    for line in block["lines"]:
                        # Get the text from the line
                        line_text = " ".join([span["text"] for span in line["spans"]])

                        # Apply ligature replacement and common word fixes
                        line_text = replace_ligatures(line_text)
                        line_text = fix_common_word_splits(line_text)

                        # **Immediately stop processing if the reference/acknowledgements section is detected**
                        if is_reference_or_acknowledgements_section(line_text):
                            section_reached = True
                            break  # Exit the inner loop and stop processing this file
                        
                        # Skip if it's a header, footnote, contains DOI/URL, or matches the title of the PDF
                        if is_heading(line_text) or is_footnote(line_text) or contains_doi_or_https(line_text) or line_text.strip().lower() == title.lower():
                            continue

                        # Get the x-coordinate (horizontal position of the first word in the line)
                        first_word_x = line["spans"][0]["bbox"][0]

                        # Check if the line belongs to the same paragraph (by horizontal position)
                        if prev_x is None or first_word_x - prev_x < 10:  # If the line's x is close to the previous, it's part of the same paragraph
                            paragraph.append(line_text)
                        else:
                            # When indentation changes significantly, treat this as the start of a new paragraph
                            if paragraph:  # If there's already accumulated text, store it as a block
                                full_paragraph_text = " ".join(paragraph).strip()
                                if count_words(full_paragraph_text) >= 10:  # Skip blocks with less than 10 words
                                    data.append([filename, page_num + 1, full_paragraph_text])
                            paragraph = [line_text]  # Start a new paragraph

                        prev_x = first_word_x  # Update the previous x-coordinate

                    # If section_reached is True after breaking, break the outer loop as well
                    if section_reached:
                        break

                    # If there's any accumulated paragraph, add it to the data
                    if paragraph and not section_reached:
                        full_paragraph_text = " ".join(paragraph).strip()
                        if count_words(full_paragraph_text) >= 10:  # Skip blocks with less than 10 words
                            data.append([filename, page_num + 1, full_paragraph_text])

# Convert the data to a DataFrame (optional)
df = pd.DataFrame(data, columns=["File", "Page", "text"])

# Print the first few records
print(df.head())


# In[4]:


# save the DataFrame to a CSV file
df.to_csv("studies_lobke.csv", index=False)


# # 2.  Load Your Data
# 
# Load the articles from your CSV file using pandas. 

# In[33]:


import pandas as pd

# Load the data
df= pd.read_csv(r'C:\Users\sadik\OneDrive\Documenten\Howest\semester6\AI_project\project\studies_lobke.csv')
df.head()


# ### Removing any personal informtion to anonymize data  

# In[34]:


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
df['text_clean'] = df['text'].apply(remove_sensitive_info)

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

# In[35]:


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
df['text_clean'] = df['text'].apply(remove_geographical_entities)

# Display a few cleaned sentences
df.head()


# In[36]:


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
minWordSize = 2

# Initialize the WordNetLemmatizer and PorterStemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preprocessing function to clean sentences
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Handle missing or non-string values

    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)

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
df['text_clean'] = df['text_clean'].apply(preprocess_text)

# Display the first elements after processing
df.head()


# In[38]:


df.info()


# ### To see how data cleaning looks 

# In[39]:


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

# In[40]:


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

# In[41]:


from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize BERTopic model
topic_model = BERTopic(calculate_probabilities=True)

# Fit the model with preprocessed text sentences
topics, probabilities = topic_model.fit_transform(df['text_clean'])

# View and inspect topics
topic_model.get_topic_info()


# In[43]:


# Initialize BERTopic model
topic_model = BERTopic(calculate_probabilities=True, min_topic_size=5, nr_topics=10)

# Fit the model with preprocessed text sentences
topics, probabilities = topic_model.fit_transform(df['text_clean'])

# View and inspect topics
topic_model.get_topic_info()



# In[44]:


topic_model.topics_[:20]


# ### Here we reduce the number of topics with the number of pdf files we have uploaded

# In[45]:


print(topic_model.topics_)


# ### Here we can search an attribute that is related to certain topics

# In[46]:


similar_topics, similarity = topic_model.find_topics("stress"); similar_topics


# In[47]:


similar_topics, similarity = topic_model.find_topics("happy"); similar_topics


# In[49]:


topic_model.get_topic(6)


# ### topic limited to the pdf count

# In[50]:


topic_model.get_topic(30)


# # 5. Visualize Topics
# We can call .visualize_topics to create a 2D representation of the topics. The  graph is a plotly interactive graph which can be converted to HTML:
# 
# Note: If you get the error 'ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed', go to terminal and type 'pip install --upgrade nbformat  ' 

# In[51]:


# Visualize topics with an interactive plot
topic_model.visualize_topics()


# You can use the slider to select the topic which then lights up red. If you hover over a topic, then general information is given about the topic, including the size of the topic and its corresponding words.
# 
# We can also ask for a representation of the corresponding words for each topic:

# In[52]:


topic_model.visualize_barchart()


# # 6. Visualize Topic Hierarchy¶
# The topics that were created can be hierarchically reduced. In order to understand the potential hierarchical structure of the topics, we can use scipy.cluster.hierarchy to create clusters and visualize how they relate to one another. We can also see what happens to the topic representations when merging topics. 

# In[54]:


hierarchical_topics = topic_model.hierarchical_topics(df['text_clean'])
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

# In[56]:


df = df.reset_index(drop=True)  # Reset index to avoid KeyError
topic_model.visualize_documents(df['text'].tolist())  # Convert Series to list


# When you hover over a point, you can see which text it is. The color tells you to which topic it belongs. While this is very pretty, it might be useful to be able to just open an excel-file or csv, which contains the original text, with the assigned topic, including the topic words:

# In[57]:


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


# In[58]:


df.head()


# We can also see the topic distribution per document = the probability that the text belongs to each topic (if a topic is not included in the graph, the probability is 0). Eg, the topic distribution for the sixth document:(!python starts counting at 0, so 6th =5)

# In[60]:


topic_model.visualize_distribution(probabilities[5])


# # 8. Topics per full article
# 
# We extract the number of times a topic is assigned within the full articles.

# In[62]:


import matplotlib.pyplot as plt

# Calculate the count of times each topic is chosen within each article
article_topic_counts = df.groupby('File')['topic_number'].value_counts().unstack(fill_value=0)

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

# In[63]:


import matplotlib.pyplot as plt

# Calculate the proportion of times each topic is chosen within each article
article_topic_proportions = df.groupby('File')['topic_number'].value_counts(normalize=True).unstack(fill_value=0)

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

# In[63]:


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

