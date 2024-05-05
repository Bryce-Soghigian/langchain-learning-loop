#!/usr/bin/env python
# coding: utf-8

# This version uses Milvus through Docker Compose so you must have Docker installed to run this notebook (Milvus is spun up via `docker compose up -d` as shown in the block below)

# In[8]:


# ! pip install -qU pymilvus langchain sentence-transformers tiktoken octoai-sdk openai
# docker-compose up -d


# In[9]:


from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OCTOAI_TOKEN"] = os.getenv("OCTOAI_API_TOKEN")


# In[10]:


from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
llm = OctoAIEndpoint(
        model="mixtral-8x22b-instruct-fp16",
        max_tokens=50000,
        presence_penalty=0,
        temperature=0.3,
        top_p=0.9,
    )


# In[11]:


from langchain_community.embeddings import OctoAIEmbeddings


# In[12]:


embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")


# In[13]:


import requests
files = []

def download_pdf(url, save_path): 
    paths = []
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        paths.append(save_path)
        f.write(response.content)

# pdf_url = "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf" 
# save_path = "cpu-intro.pdf"
# download_pdf(pdf_url, save_path)


# In[14]:


import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus

files = ["laws_of_motion.pdf"]
# Initialize the embeddings API
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

def split_large_chunks(text, max_size=1000):
    """Split large chunks into smaller parts that comply with the size limit."""
    parts = []
    while len(text) > max_size:
        part = text[:max_size]
        parts.append(part)
        text = text[max_size:]
    if text:
        parts.append(text) 
    return parts

def process_file(file_path):
    file_texts = []
    with pdfplumber.open(file_path) as pdf:
        all_text = []
        for page in pdf.pages:
            page_text = page.extract_text()  
            if page_text:
                all_text.append(page_text)

        combined_text = "\n".join(all_text)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=8000, chunk_overlap=240
        )
        texts = text_splitter.split_text(combined_text)
        for i, chunked_text in enumerate(texts):
            if len(chunked_text) > 10240:
                print(f"Chunk too large, further splitting: {len(chunked_text)} characters")
                smaller_chunks = split_large_chunks(chunked_text)
                for j, small_chunk in enumerate(smaller_chunks):
                    if len(small_chunk) > 10240:
                        print(f"Error: Sub-chunk still too large: {len(small_chunk)} characters")
                        continue
                    file_texts.append(Document(page_content=small_chunk,
                        metadata={"doc_title": file_path.split(".")[0], "chunk_num": f"{i}-{j}"}))
            else:
                file_texts.append(Document(page_content=chunked_text,
                        metadata={"doc_title": file_path.split(".")[0], "chunk_num": i}))
    return file_texts

all_documents = []
for file in files:
    all_documents.extend(process_file(file))

try:
    # Ensure no document exceeds the length limit
    for doc in all_documents:
        if len(doc.page_content) > 10240:
            raise ValueError(f"Document exceeds max length: {len(doc.page_content)} characters")
    vector_store = Milvus.from_documents(
        all_documents,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": 19530},
        collection_name="motion10"
    )
except ValueError as e:
    print(f"Failed to create embeddings due to an error: {e}")


# In[ ]:


retriever = vector_store.as_retriever()


# In[ ]:


from langchain.prompts import ChatPromptTemplate
template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)


# 

# In[ ]:


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



# In[ ]:


import json

def quiz_loop(questions):
    score = 0
    missed_topics = {}

    for key, question in questions.items():
        print(f"Question: {question}")
        answer = input("Enter your answer (or type 'skip' to pass): ")
        if answer.lower() == 'skip':
            missed_topics[key] = question
            continue

        correctness = chain.invoke(f"score the response based on the accuracy of the input. output a python bool if answer: {answer} is a correct answer for the question: {question} output only 'True' if true, output only 'False' if false")
        print(correctness)
        if correctness == "True":
            score += 1
        else:
            missed_topics[key] = question

    return score, missed_topics


missedTopics = ""

while True:
    prompt = "Based on the content of the provided chapter, generate a list of two detailed questions that cover key concepts and themes. ask important questions that matter instead of questions on minute details that are not relevant to the main concepts. These concepts and themes must be returned in json with a key: value format. after the two responses are provided, you can ask me two more questions and continue the process. "
    questions_json = chain.invoke(prompt)
    questions = json.loads(questions_json)

    print("\nStarting the quiz...")
    score, missed_topics = quiz_loop(questions)
    print(f"Initial score: {score}/{len(questions)}")
    
    if missed_topics:
        print("\nYou missed some questions. Let's try those again.")
        score, _ = quiz_loop(missed_topics)
        print(f"Re-quiz score: {score}/{len(missed_topics)}")

    continue_quiz = input("Do you want to continue studying? (yes/no): ").lower()
    if continue_quiz != 'yes':
        break


# In[ ]:




