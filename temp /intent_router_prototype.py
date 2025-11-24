import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from llama_cpp import Llama

# Configuration
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_PATH = os.path.expanduser("~/.node-llama-cpp/models/hf_Qwen_Qwen3-4B.Q8_0.gguf")

print(f"Checking for local model at: {LOCAL_LLM_PATH}")
if os.path.exists(LOCAL_LLM_PATH):
    print("Local model found.")
else:
    print("WARNING: Local model NOT found. Please check the path.")

# 1. Define Prototypes
prototypes = {
    "legal": [
        "Analyze this complaint for jurisdictional deficiencies under FRCP 12(b)(2).",
        "Draft a motion to compel discovery regarding the withheld email communications.",
        "Summarize the precedent set by International Shoe regarding minimum contacts.",
        "Review this lease agreement for any clauses that violate local rent control ordinances.",
        "File a motion to dismiss",
        "Plaintiff's complaint",
        "Draft a contract for services"
    ],
    "coding": [
        "Write a Python script to parse PDF court dockets and extract filing dates.",
        "Debug this SQL query that joins the case_metadata and rulings tables.",
        "Write a function",
        "Debug code",
        "API endpoint",
        "SQL query"
    ],
    "general": [
        "Draft an email to the client scheduling a meeting.",
        "What is the capital of France?",
        "Hello",
        "How are you",
        "What is the weather"
    ]
}

keyword_map = {
    "contract": "legal",
    "motion": "legal",
    "lawsuit": "legal",
    "litigation": "legal",
    "plaintiff": "legal",
    "defendant": "legal",
    "function": "coding",
    "debug": "coding",
    "api": "coding",
    "sql": "coding"
}

# 2. Model Loading & Embedding Functions
print("Loading Legal BERT...")
tokenizer_bert = AutoTokenizer.from_pretrained(LEGAL_BERT_MODEL)
model_bert = AutoModel.from_pretrained(LEGAL_BERT_MODEL)

print("Loading MiniLM...")
model_minilm = SentenceTransformer(MINILM_MODEL)

def get_embedding_bert(text):
    inputs = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def get_embedding_minilm(text):
    return model_minilm.encode(text)

# Pre-compute prototype embeddings
print("Encoding prototypes...")
prototype_embeddings_bert = {k: [get_embedding_bert(p) for p in v] for k, v in prototypes.items()}
prototype_embeddings_minilm = {k: [get_embedding_minilm(p) for p in v] for k, v in prototypes.items()}
print("Models loaded and prototypes encoded.")

# 3. Hybrid Router Logic
def calculate_confidence(query_embedding, prototype_embeddings_map):
    max_sim = -1
    best_intent = "general"
    
    for intent, protos in prototype_embeddings_map.items():
        for proto_emb in protos:
            sim = cosine_similarity([query_embedding], [proto_emb])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_intent = intent
                
    return best_intent, max_sim

def keyword_search(text):
    lower = text.lower()
    for keyword, intent in keyword_map.items():
        if keyword in lower:
            return intent
    return None

def route_query(text, model_type="bert"):
    t0 = time.perf_counter()
    
    # Layer 1: Keyword
    keyword_intent = keyword_search(text)
    
    # Layer 2: Vector
    t1 = time.perf_counter()
    if model_type == "bert":
        embedding = get_embedding_bert(text)
        proto_map = prototype_embeddings_bert
    else:
        embedding = get_embedding_minilm(text)
        proto_map = prototype_embeddings_minilm
    t2 = time.perf_counter()
    
    vector_intent, vector_score = calculate_confidence(embedding, proto_map)
    t3 = time.perf_counter()
    
    # Hybrid Logic
    final_intent = vector_intent
    handler = "semantic-only"
    confidence = vector_score
    
    if keyword_intent:
        # If keyword matches vector, boost confidence
        if keyword_intent == vector_intent:
             if vector_score > 0.6:
                confidence = 1.0
                handler = "keyword-confirmed"
        # If keyword contradicts vector
        else:
            if vector_score < 0.4:
                # Trust vector if score is very low (false positive keyword?)
                pass
            else:
                # Ambiguous, maybe lean towards vector but log it
                pass
            
    if confidence < 0.7 and handler != "keyword-confirmed":
        handler = "fallback"
        final_intent = "general"
        
    timing = {
        "total": (t3 - t0) * 1000,
        "embedding": (t2 - t1) * 1000,
        "similarity": (t3 - t2) * 1000
    }
    
    return {
        "intent": final_intent,
        "confidence": confidence,
        "handler": handler,
        "timing_ms": timing,
        "model": model_type
    }

# 4. Benchmarking & Testing
test_queries = [
    "Draft a motion to dismiss for lack of jurisdiction",
    "Write a python script to parse PDF court dockets",
    "What is the weather like today?",
    "Review this NDA for me",
    "debug this sql error in the users table",
    "Can you help me find a good restaurant?",
    "Summarize the facts of the case"
]

print(f"\n{'Query':<50} | {'Intent':<10} | {'Conf':<5} | {'Handler':<15} | {'Time(ms)':<10}")
print("-" * 100)

for query in test_queries:
    # Test BERT
    res = route_query(query, model_type="bert")
    print(f"{query[:47]+'...':<50} | {res['intent']:<10} | {res['confidence']:.2f}  | {res['handler']:<15} | {res['timing_ms']['total']:.2f} (BERT)")
    
    # Test MiniLM
    res = route_query(query, model_type="minilm")
    print(f"{'':<50} | {res['intent']:<10} | {res['confidence']:.2f}  | {res['handler']:<15} | {res['timing_ms']['total']:.2f} (MiniLM)")
    print("-" * 100)

# 5. End-to-End Chatbot Simulation (Local Qwen)
def run_chatbot_simulation(query):
    # 1. Route
    route_result = route_query(query, model_type="bert")
    intent = route_result["intent"]
    print(f"\nUser Query: {query}")
    print(f"Detected Intent: {intent} (Confidence: {route_result['confidence']:.2f})")
    
    if not os.path.exists(LOCAL_LLM_PATH):
        print("Skipping LLM generation (Model not found)")
        return

    # 2. Load LLM
    try:
        print("Loading Local LLM...")
        llm = Llama(
            model_path=LOCAL_LLM_PATH,
            n_ctx=2048,
            verbose=False
        )
        
        # 3. Prompt Engineering based on Intent
        system_prompt = "You are a helpful assistant."
        if intent == "legal":
            system_prompt = "You are an expert legal assistant. Provide precise, professional legal information. Cite sources if possible."
        elif intent == "coding":
            system_prompt = "You are an expert software engineer. Provide clean, efficient, and well-documented code."
            
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        # 4. Generate
        print("Generating response...")
        output = llm(
            prompt,
            max_tokens=128, # Short generation for test
            stop=["<|im_end|>"],
            echo=False
        )
        
        print("Response:")
        print(output['choices'][0]['text'])
        
    except Exception as e:
        print(f"Error running LLM: {e}")

# Run simulation
run_chatbot_simulation("Draft a motion to dismiss for a case of defamation charged by a strore wanderer, when a retailer defamed him as a thief. ")
