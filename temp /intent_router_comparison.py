#!/usr/bin/env python3
"""
Intent Router Comparison Suite
Compares 3 routing approaches with llama.cpp inference
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ML/NLP Libraries
import numpy as np
from sentence_transformers import SentenceTransformer

# Semantic Router (optional - will check if installed)
try:
    from semantic_router import Route
    from semantic_router.routers import HybridRouter as SemanticHybridRouter
    from semantic_router.encoders import HuggingFaceEncoder, BM25Encoder
    SEMANTIC_ROUTER_AVAILABLE = True
except ImportError:
    SEMANTIC_ROUTER_AVAILABLE = False
    print("Warning: semantic-router not installed. Skipping semantic-router tests.")

# Llama.cpp
from llama_cpp import Llama


# ============================================================================
# TEST QUERIES (from your logs)
# ============================================================================
TEST_QUERIES = [
    {"id": 1, "text": "hi", "expected_intent": "general"},
    {
        "id": 2,
        "text": "Draft a motion to dismiss for a case of defamation charged by a strore wanderer, when a retailer defamed him as a thief.",
        "expected_intent": "legal"
    },
    {
        "id": 3,
        "text": "I want to sue my friend. what can i do?",
        "expected_intent": "legal"
    },
    {
        "id": 4,
        "text": "I want to draft a motion to dismiss for lack of jurisdiction.",
        "expected_intent": "legal"
    },
    {
        "id": 5,
        "text": "Draft a motion to compel deffamation regarding the withheld email communications that i recieved. I think it falls under FRCP 12(b)(2)",
        "expected_intent": "legal"
    },
    {
        "id": 6,
        "text": "I want to learn python.",
        "expected_intent": "coding"
    }
]


# ============================================================================
# PROTOTYPES (from your IntentRouter.ts)
# ============================================================================
PROTOTYPES = {
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


# ============================================================================
# ROUTER 1: CURRENT SYSTEM (from your IntentRouter.ts)
# ============================================================================
class CurrentRouter:
    """Port of your TypeScript IntentRouter logic"""
    
    def __init__(self):
        print("Initializing CurrentRouter...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.keyword_map = {
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
        
        # Pre-compute prototype embeddings
        self.prototype_embeddings = {}
        for intent, examples in PROTOTYPES.items():
            self.prototype_embeddings[intent] = []
            for example in examples:
                emb = self.encoder.encode(example, normalize_embeddings=True)
                self.prototype_embeddings[intent].append(emb)
        
        print(f"  Loaded {sum(len(v) for v in self.prototype_embeddings.values())} prototypes")
    
    def route(self, text: str) -> Dict:
        t0 = time.perf_counter()
        
        # Layer 1: Keyword search
        t_kw_start = time.perf_counter()
        keyword_intent = self._check_keywords(text)
        t_kw = (time.perf_counter() - t_kw_start) * 1000
        
        # Layer 2: Embedding
        t_emb_start = time.perf_counter()
        query_emb = self.encoder.encode(text, normalize_embeddings=True)
        t_emb = (time.perf_counter() - t_emb_start) * 1000
        
        # Layer 3: Similarity
        t_sim_start = time.perf_counter()
        vector_intent, vector_score = self._compute_best_match(query_emb)
        t_sim = (time.perf_counter() - t_sim_start) * 1000
        
        # Hybrid logic (exact replica of your TS code)
        confidence = vector_score
        handler = "semantic-only"
        final_intent = vector_intent
        
        if keyword_intent:
            if keyword_intent == vector_intent and vector_score > 0.6:
                confidence = 1.0
                handler = "keyword-confirmed"
            elif vector_score < 0.4:
                confidence = vector_score
                handler = "semantic-only"
            else:
                confidence = vector_score
                handler = "semantic-only"
        
        # Fallback check
        if confidence < 0.7 and handler != "keyword-confirmed":
            handler = "fallback"
            final_intent = "general"
        
        total_time = (time.perf_counter() - t0) * 1000
        
        return {
            "intent": final_intent,
            "confidence": float(confidence),
            "handler": handler,
            "time_ms": total_time,
            "breakdown_ms": {
                "keyword": t_kw,
                "embedding": t_emb,
                "similarity": t_sim
            }
        }
    
    def _check_keywords(self, text: str) -> str:
        lower = text.lower()
        for keyword, intent in self.keyword_map.items():
            if keyword in lower:
                return intent
        return None
    
    def _compute_best_match(self, query_emb) -> Tuple[str, float]:
        max_score = -1
        best_intent = "general"
        
        for intent, embeddings in self.prototype_embeddings.items():
            for emb in embeddings:
                score = np.dot(query_emb, emb)  # Cosine similarity (normalized)
                if score > max_score:
                    max_score = score
                    best_intent = intent
        
        return best_intent, float(max_score)


# ============================================================================
# ROUTER 2: SEMANTIC-ROUTER (official library)
# ============================================================================
class SemanticRouterWrapper:
    """Wrapper for semantic-router library"""
    
    def __init__(self):
        if not SEMANTIC_ROUTER_AVAILABLE:
            raise ImportError("semantic-router not installed")
        
        print("Initializing SemanticRouterWrapper...")
        
        # Create routes from prototypes
        routes = []
        for intent, utterances in PROTOTYPES.items():
            routes.append(Route(name=intent, utterances=utterances))
        
        # Initialize hybrid router
        self.router = SemanticHybridRouter(
            encoder=HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2"),
            sparse_encoder=BM25Encoder(),
            routes=routes,
            alpha=0.3  # 30% dense, 70% sparse
        )
        
        print(f"  Initialized with alpha={self.router.alpha}")
    
    def route(self, text: str) -> Dict:
        t0 = time.perf_counter()
        
        result = self.router(text)
        
        total_time = (time.perf_counter() - t0) * 1000
        
        return {
            "intent": result.name if result else "general",
            "confidence": float(getattr(result, 'similarity_score', 0.0)) if result else 0.0,
            "handler": "semantic-router-hybrid",
            "time_ms": total_time,
            "breakdown_ms": {}
        }


# ============================================================================
# ROUTER 3: HYBRID MODIFIED (your system + semantic-router concepts)
# ============================================================================
class HybridModifiedRouter:
    """Enhanced version with BM25 + lowered thresholds + expanded keywords"""
    
    def __init__(self):
        print("Initializing HybridModifiedRouter...")
        
        # Dense encoder
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Expanded keyword map
        self.keyword_map = {
            # Legal
            "contract": "legal", "motion": "legal", "lawsuit": "legal",
            "litigation": "legal", "plaintiff": "legal", "defendant": "legal",
            "sue": "legal", "defamation": "legal", "court": "legal",
            "jurisdiction": "legal",
            # Coding
            "function": "coding", "debug": "coding", "api": "coding",
            "sql": "coding", "python": "coding", "code": "coding",
            "script": "coding"
        }
        
        # Alpha for convex scaling
        self.alpha = 0.3
        
        # Pre-compute embeddings
        self.prototype_embeddings = {}
        for intent, examples in PROTOTYPES.items():
            self.prototype_embeddings[intent] = []
            for example in examples:
                emb = self.encoder.encode(example, normalize_embeddings=True)
                self.prototype_embeddings[intent].append(emb)
        
        # Simple keyword scoring (instead of full BM25)
        self.keyword_scores = self._build_keyword_scores()
        
        print(f"  Alpha={self.alpha}, Keywords={len(self.keyword_map)}")
    
    def _build_keyword_scores(self) -> Dict:
        """Build keyword importance scores from prototypes"""
        scores = {intent: {} for intent in PROTOTYPES.keys()}
        
        for intent, examples in PROTOTYPES.items():
            for keyword, keyword_intent in self.keyword_map.items():
                if keyword_intent == intent:
                    # Count occurrences in prototypes
                    count = sum(1 for ex in examples if keyword in ex.lower())
                    scores[intent][keyword] = min(1.0, count / len(examples) + 0.5)
        
        return scores
    
    def route(self, text: str) -> Dict:
        t0 = time.perf_counter()
        
        # Layer 1: Keyword detection
        t_kw_start = time.perf_counter()
        keyword_matches = self._check_keywords_multi(text)
        t_kw = (time.perf_counter() - t_kw_start) * 1000
        
        # Layer 2a: Dense embedding
        t_emb_start = time.perf_counter()
        query_emb = self.encoder.encode(text, normalize_embeddings=True)
        t_emb = (time.perf_counter() - t_emb_start) * 1000
        
        # Layer 2b: Dense similarity
        t_sim_start = time.perf_counter()
        dense_intent, dense_score = self._compute_best_match(query_emb)
        
        # Layer 2c: Keyword-based scoring (sparse proxy)
        sparse_scores = self._compute_keyword_scores(text, keyword_matches)
        sparse_intent = max(sparse_scores, key=sparse_scores.get)
        sparse_score = sparse_scores[sparse_intent]
        t_sim = (time.perf_counter() - t_sim_start) * 1000
        
        # Convex scaling fusion
        if dense_intent == sparse_intent:
            # Agreement - use weighted combo
            final_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            final_intent = dense_intent
            handler = "hybrid-consensus"
        elif keyword_matches and keyword_matches[0][1] == dense_intent:
            # Keyword confirms dense
            final_score = 0.5 * dense_score + 0.5 * keyword_matches[0][2]
            final_intent = dense_intent
            handler = "keyword-dense-confirm"
        else:
            # Disagreement - weighted combo anyway
            final_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            final_intent = dense_intent if dense_score > sparse_score else sparse_intent
            handler = "weighted-choice"
        
        # Lowered threshold (0.5 instead of 0.7)
        if final_score < 0.5:
            final_intent = "general"
            handler = "fallback"
        
        total_time = (time.perf_counter() - t0) * 1000
        
        return {
            "intent": final_intent,
            "confidence": float(final_score),
            "handler": handler,
            "time_ms": total_time,
            "breakdown_ms": {
                "keyword": t_kw,
                "embedding": t_emb,
                "similarity": t_sim
            },
            "debug": {
                "dense_score": float(dense_score),
                "sparse_score": float(sparse_score),
                "dense_intent": dense_intent,
                "sparse_intent": sparse_intent
            }
        }
    
    def _check_keywords_multi(self, text: str) -> List[Tuple[str, str, float]]:
        """Returns list of (keyword, intent, score) matches"""
        lower = text.lower()
        matches = []
        for keyword, intent in self.keyword_map.items():
            if keyword in lower:
                score = self.keyword_scores.get(intent, {}).get(keyword, 0.8)
                matches.append((keyword, intent, score))
        return matches
    
    def _compute_keyword_scores(self, text: str, keyword_matches: List) -> Dict[str, float]:
        """Compute intent scores based on keyword matches"""
        scores = {intent: 0.0 for intent in PROTOTYPES.keys()}
        
        if not keyword_matches:
            return scores
        
        # Aggregate scores by intent
        for keyword, intent, score in keyword_matches:
            scores[intent] += score
        
        # Normalize
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1.0
        for intent in scores:
            scores[intent] = scores[intent] / max_score
        
        return scores
    
    def _compute_best_match(self, query_emb) -> Tuple[str, float]:
        max_score = -1
        best_intent = "general"
        
        for intent, embeddings in self.prototype_embeddings.items():
            for emb in embeddings:
                score = np.dot(query_emb, emb)
                if score > max_score:
                    max_score = score
                    best_intent = intent
        
        return best_intent, float(max_score)


# ============================================================================
# LLAMA.CPP INFERENCE
# ============================================================================
class LlamaInference:
    """Wrapper for llama.cpp inference"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.expanduser("~/.node-llama-cpp/models/hf_Qwen_Qwen3-4B.Q8_0.gguf")
        
        print(f"Loading Llama model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            verbose=False
        )
        print("  Model loaded")
    
    def generate(self, query: str, intent: str, max_tokens: int = 100) -> Dict:
        """Generate response with timing metrics"""
        
        # System prompts by intent
        system_prompts = {
            "legal": "You are an expert legal assistant. Provide precise, professional legal information.",
            "coding": "You are an expert software engineer. Provide clean, efficient code.",
            "general": "You are a helpful assistant."
        }
        
        prompt = f"<|im_start|>system\n{system_prompts[intent]}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        t0 = time.perf_counter()
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|im_end|>"],
            echo=False
        )
        
        total_time = (time.perf_counter() - t0) * 1000
        
        response_text = output['choices'][0]['text'].strip()
        
        # Rough TTFT estimate (llama.cpp doesn't expose directly)
        # Assume linear generation, TTFT ≈ total_time / tokens * 1.5
        usage = output.get('usage', {})
        tokens_generated = usage.get('completion_tokens', len(response_text.split()))
        ttft_estimate = (total_time / max(tokens_generated, 1)) * 1.5 if tokens_generated > 0 else total_time
        
        return {
            "response": response_text,
            "total_time_ms": total_time,
            "ttft_ms": ttft_estimate,
            "tokens": tokens_generated
        }


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def run_comparison():
    """Main comparison suite"""
    
    print("\n" + "="*70)
    print("INTENT ROUTER COMPARISON SUITE")
    print("="*70)
    
    # Initialize routers
    routers = {}
    
    print("\n[1/3] Initializing CurrentRouter...")
    routers["current"] = CurrentRouter()
    
    if SEMANTIC_ROUTER_AVAILABLE:
        print("\n[2/3] Initializing SemanticRouterWrapper...")
        try:
            routers["semantic"] = SemanticRouterWrapper()
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n[2/3] Skipping SemanticRouterWrapper (not installed)")
    
    print("\n[3/3] Initializing HybridModifiedRouter...")
    routers["hybrid_modified"] = HybridModifiedRouter()
    
    # Initialize Llama
    print("\n" + "-"*70)
    llama = LlamaInference()
    
    # Run tests
    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70)
    
    results = []
    
    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"Query {query['id']}: {query['text'][:60]}...")
        print(f"Expected: {query['expected_intent']}")
        print('-'*70)
        
        query_results = {
            "query_id": query['id'],
            "text": query['text'],
            "expected_intent": query['expected_intent'],
            "routers": {}
        }
        
        for router_name, router in routers.items():
            print(f"\n[{router_name.upper()}]")
            
            # Route
            route_result = router.route(query['text'])
            correct = route_result['intent'] == query['expected_intent']
            
            print(f"  Intent: {route_result['intent']} {'✅' if correct else '❌'}")
            print(f"  Confidence: {route_result['confidence']:.3f}")
            print(f"  Handler: {route_result['handler']}")
            print(f"  Route Time: {route_result['time_ms']:.2f}ms")
            
            if 'debug' in route_result:
                print(f"  Debug: dense={route_result['debug']['dense_score']:.3f}, sparse={route_result['debug']['sparse_score']:.3f}")
            
            # Generate response
            gen_result = llama.generate(query['text'], route_result['intent'], max_tokens=100)
            print(f"  Generation: {gen_result['total_time_ms']:.2f}ms ({gen_result['tokens']} tokens)")
            print(f"  TTFT (est): {gen_result['ttft_ms']:.2f}ms")
            print(f"  Response: {gen_result['response'][:80]}...")
            
            query_results['routers'][router_name] = {
                "routing": route_result,
                "generation": gen_result,
                "correct": correct
            }
        
        results.append(query_results)
    
    # Save and report
    save_results(results, routers.keys())
    
    return results


def save_results(results: List[Dict], router_names: List[str]):
    """Save results to JSON and Markdown"""
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate Markdown
    md = generate_markdown_report(results, router_names)
    md_path = output_dir / "comparison_report.md"
    with open(md_path, "w") as f:
        f.write(md)
    
    print("\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def generate_markdown_report(results: List[Dict], router_names: List[str]) -> str:
    """Generate markdown summary"""
    
    md = "# Intent Router Comparison Report\n\n"
    md += f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"**Test Queries**: {len(results)}\n\n"
    
    # Accuracy table
    md += "## Accuracy\n\n"
    md += "| Router | Correct | Incorrect | Accuracy |\n"
    md += "|--------|---------|-----------|----------|\n"
    
    for router_name in router_names:
        correct = sum(1 for r in results if r['routers'][router_name]['correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        md += f"| {router_name} | {correct} | {total - correct} | {accuracy:.1%} |\n"
    
    # Performance table
    md += "\n## Performance (Average)\n\n"
    md += "| Router | Intent Detection (ms) | TTFT (ms) | Total Generation (ms) |\n"
    md += "|--------|----------------------|-----------|----------------------|\n"
    
    for router_name in router_names:
        avg_route = sum(r['routers'][router_name]['routing']['time_ms'] for r in results) / len(results)
        avg_ttft = sum(r['routers'][router_name]['generation']['ttft_ms'] for r in results) / len(results)
        avg_gen = sum(r['routers'][router_name]['generation']['total_time_ms'] for r in results) / len(results)
        md += f"| {router_name} | {avg_route:.2f} | {avg_ttft:.2f} | {avg_gen:.2f} |\n"
    
    # Per-query breakdown
    md += "\n## Per-Query Results\n\n"
    for result in results:
        md += f"\n### Query {result['query_id']}\n\n"
        md += f"**Text**: {result['text']}\n\n"
        md += f"**Expected Intent**: `{result['expected_intent']}`\n\n"
        md += "| Router | Predicted | Confidence | Correct | Route Time (ms) |\n"
        md += "|--------|-----------|------------|---------|------------------|\n"
        
        for router_name in router_names:
            r = result['routers'][router_name]
            check = "✅" if r['correct'] else "❌"
            md += f"| {router_name} | {r['routing']['intent']} | {r['routing']['confidence']:.3f} | {check} | {r['routing']['time_ms']:.2f} |\n"
    
    return md


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        results = run_comparison()
        print("\n✅ Comparison complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
