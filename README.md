# RAG Workshop Guide 🚀

Welcome to the **Retrieval-Augmented Generation (RAG) Workshop**! This hands-on workshop will guide you through building an end-to-end RAG pipeline, covering everything from foundational concepts to practical implementation.

## 🎯 Workshop Goals
> - Develop an understanding of how RAG improves generative AI with external knowledge.
> - Learn how retrieval augments generative models to improve in many ways!
> - Experiment with provided scripts to ground theory.
> - Build a simple yet functional RAG pipeline using Python, Ollama, and ChromaDB.
> - Dive into advanced topics and optimization techniques in the last session!

---

## 📜 Table of Contents

1. [Introduction & Workshop Overview](#introduction--workshop-overview)
2. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
3. [Session 1: Understanding RAG and Embedding Concepts](#session-1-understanding-rag-and-embedding-concepts)
4. [Session 2: RAG Applied](#session-2-rag-mechanics-and-initial-exploration)
5. [Session 3: My first RAG](#session-3-building-your-rag-pipeline)
6. [Session 4: Deep Learning & Advanced RAG](#session-4-deep-learning--advanced-rag)
7. [Additional Resources & Contributing](#additional-resources--contributing)

---

## 📖 Introduction & Workshop Overview

This workshop introduces the **Retrieval-Augmented Generation (RAG)** system, demonstrating how retrieval mechanisms enhance generative models. You'll gain hands-on experience building a RAG pipeline with minimal complexity.

### Who Should Attend?
- Entry-level engineers and AI enthusiasts with basic Python knowledge.
- Developers interested in integrating retrieval into generative AI models.

### Expected Outcome
By the end, you’ll have a functional RAG system retrieving relevant knowledge for enhanced AI-generated responses.

---

## ⚙️ Prerequisites & Environment Setup

### Required Software & Tools
- **Python 3.10+** ([Download Python](https://www.python.org/))
- **Ollama** ([Download Ollama](https://ollama.com/))
- **Hardware:** Minimum 8GB RAM, 4 vCores @ 2.5 GHz
- **Dependencies:** Install via:
  ```bash
  pip install -r requirements.txt
  ```
- **Key Libraries:**
  - `transformers` - For working with pre-trained language models.
  - `sentence-transformers` - To generate embeddings for text retrieval.
  - `chromadb` - A lightweight vector database for document retrieval.
  - `faiss-cpu` or `faiss-gpu` (optional) - For fast vector searches.
  - `nltk` - For basic text preprocessing and tokenization.
  - `ollama` - For model interaction and response generation.

### Initial Setup Check
Run the following command to verify installation:
```bash
python testollama.py
```
**Expected Outcome:** This script loads a *local* model and allows you to chat with it.

**Trouble shooting:** If requirements.txt does not install chromadb, [check this](https://stackoverflow.com/questions/76856170/error-failed-building-wheel-for-chroma-hnswlib-trying-to-install-chromadb-on)

---

## 📚 Session 1: Understanding RAG and Embedding Concepts

### 1.1 Deep Learning Basics
- Overview of Deep Learning vs other AI fields.
- Key differences between traditional ML and deep learning models.

### 1.2 RAG Fundamentals
- Understanding why retrieval is essential in generative AI.
- How RAG combines retrieval and generation to enhance responses.

#### Exercise: Test Baseline LLM Generation
Run the testollama script to generate text:
```bash
python testollama.py
```
**Expected Outcome:** Observe the limitations of traditional LLM outputs.

### 1.3 Understanding Embeddings
- What are embeddings, and why do they matter in retrieval-based AI?
- How embeddings represent semantic meaning in vector space.

#### Exercise: Visualizing Embeddings
Run visualization tool:
```bash
python viz_embeddings.py
```
**Expected Outcome:** 3D visualization of word embeddings and their relationships.

---

## 🤖 Session 2: RAG Mechanics and Initial Exploration

### 2.1 RAG Components Breakdown
- Overview of the major components of a RAG pipeline:
  - Query encoding
  - Document retrieval
  - Response generation
- Understanding different retrieval mechanisms (keyword search vs. vector search).

### 2.2 Implementing a Simple RAG Pipeline
- Hands-on example of a basic RAG implementation.
- Example repository: [GitHub Repo](https://github.com/mariomanza150/rag_data_extraction.git)

#### Exercise: Modify Retrieval Behavior
Adjust the number of retrieved documents in `retrieve_documents`:
```python
def retrieve_documents(query_embedding, k=3):
    retrieved_docs = chroma_db.query(query_embedding, top_k=k)
    return retrieved_docs
```
**Expected Outcome:** Analyze how retrieval size impacts the final generated response.

---

## 💻 Session 3: Building Your RAG Pipeline

### 3.1 Data Preparation
- Preparing datasets for use in RAG-based retrieval.
- Tokenization and text preprocessing strategies.

#### Exercise: Run Data Preprocessing
Execute preprocessing script:
```bash
python prepare_data.py
```
**Expected Outcome:** Saves tokenized sentences in `sentences.json`.

### 3.2 Embedding Generation
- Converting textual data into numerical embeddings.
- How sentence embeddings impact retrieval quality.

#### Exercise: Print Embedding Shape
Modify `viz_embeddings.py`:
```python
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
```
Run:
```bash
python viz_embeddings.py
```
**Expected Outcome:** Outputs the dimensionality of generated embeddings.

### 3.3 Retrieval & Response Generation
- Integrating document retrieval into generative AI.
- Fine-tuning retrieval parameters for better results.

#### Exercise: Implement Augmented Response Generation
Modify `testollama.py`:
```python
def generate_augmented_response(query):
    query_embedding = model.encode([query])[0]
    docs = retrieve_documents(query_embedding, k=5)
    augmented_query = query + "\n" + "\n".join([doc for doc, _ in docs])
    response = ollama.generate(augmented_query)
    return response
```
Run:
```bash
python testollama.py
```
**Expected Outcome:** Compare baseline vs. RAG-enhanced responses.

---

## 🔍 Session 4: Deep Learning & Advanced RAG

### 4.1 Deep Learning 101
- Introduction to neural networks and their significance.

#### 4.1.1 The Perceptron: Building Block of Neural Networks
- Explanation of perceptron architecture and activation functions.
- How perceptrons are combined into multi-layer networks.
- brief look at how backpropagation works, how 'learning' happens

#### 4.1.2 Simple Neural Network Example
- High-level overview of Convolutional Neural Networks (CNNs) for vision tasks.
- Introduction to Recurrent Neural Networks (RNNs) for sequential data processing.

#### Exercise: Implement a Simple Neural Network
Run a basic neural network training script:
```bash
python simple_nn.py
```
**Expected Outcome:** Observe the training process and model performance.

### 4.2 Transformers: The Building Block of LLMs
- How transformers revolutionized NLP and generative AI.
- The self-attention mechanism and its impact on model performance.

#### Exercise: Visualizing Attention Mechanisms
```bash
python advanced_attention.py
```
**Expected Outcome:** Understand how attention layers process input.

### 4.3 RAG 200: Deeper Component Analysis & Caveats
- Detailed breakdown of RAG’s retrieval and generation mechanisms.
- Challenges such as retrieval latency and hallucinations.

#### Exercise: Measure Retrieval Performance
Modify retrieval function:
```python
def timed_retrieve(query_embedding, k=5):
    start_time = time.time()
    retrieved_docs = chroma_db.query(query_embedding, top_k=k)
    elapsed_time = time.time() - start_time
    print(f"Retrieval took {elapsed_time:.2f} seconds")
    return retrieved_docs
```
Run:
```bash
python rag_pipeline.py --benchmark
```
**Expected Outcome:** Evaluate retrieval efficiency and potential bottlenecks.

---

## 📖 Additional Resources & Contributing

### Additional Resources
- **Hugging Face RAG Guide:** [Read More](https://huggingface.co/blog/rag)
- **Attention Is All You Need (Paper):** [ArXiv Link](https://arxiv.org/abs/1706.03762)
- **Deep Learning Book:** [Read More](https://www.deeplearningbook.org/)

### Contributing
We welcome contributions! Open an issue or submit a pull request to enhance this workshop.
