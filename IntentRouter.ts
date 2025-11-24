import { pipeline } from '@xenova/transformers';
import { logger } from './Logger.js';
import { performance } from 'perf_hooks';

export interface RouterResponse {
    intent: string;
    confidence: number;
    adapter_id: string;
    handler: string;
    timing_ms: {
        keyword_search: number;
        embedding_compute: number;
        similarity_compute: number;
        total: number;
    };
}

export class IntentRouter {
    private classifier: any = null;

    // Expanded Keyword Map (added sue, defamation, court, jurisdiction, python, code, script)
    private keywordMap: Record<string, string> = {
        // Legal domain
        "contract": "legal",
        "motion": "legal",
        "lawsuit": "legal",
        "litigation": "legal",
        "plaintiff": "legal",
        "defendant": "legal",
        "sue": "legal",
        "defamation": "legal",
        "court": "legal",
        "jurisdiction": "legal",
        // Coding domain
        "function": "coding",
        "debug": "coding",
        "api": "coding",
        "sql": "coding",
        "python": "coding",
        "code": "coding",
        "script": "coding"
    };

    // Prototypes from Plan
    private prototypes: Record<string, string[]> = {
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
    };

    private prototypeEmbeddings: Record<string, any[]> = {};
    
    // Alpha parameter for convex scaling (30% dense, 70% sparse)
    private alpha: number = 0.3;

    constructor() { }

    async initialize() {
        const t0 = performance.now();
        logger.info('SYS', 'Initializing IntentRouter...');

        // Load the embedding model
        // Using all-MiniLM-L6-v2 as per plan (Legal BERT requires different tokenizer handling in JS)
        this.classifier = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

        // Pre-compute embeddings for prototypes
        for (const [intent, examples] of Object.entries(this.prototypes)) {
            this.prototypeEmbeddings[intent] = [];
            for (const example of examples) {
                const output = await this.classifier(example, { pooling: 'mean', normalize: true });
                this.prototypeEmbeddings[intent].push(output.data);
            }
        }

        const t1 = performance.now();
        logger.info('SYS', 'IntentRouter initialized', {
            init_time_ms: t1 - t0,
            prototype_count: Object.values(this.prototypes).flat().length
        });
    }

    async route(text: string, requestId: string): Promise<RouterResponse> {
        const t_start = performance.now();

        // Layer 1: Keyword Search (Multi-keyword detection)
        const t0 = performance.now();
        const keywordMatches = this.checkKeywordsMulti(text);
        const t1 = performance.now();

        // Layer 2: Embedding Computation
        const output = await this.classifier(text, { pooling: 'mean', normalize: true });
        const queryEmbedding = output.data;
        const t2 = performance.now();

        // Layer 3a: Dense Similarity Computation
        const { bestIntent: denseIntent, maxScore: denseScore } = this.computeBestMatch(queryEmbedding);
        
        // Layer 3b: Sparse Score Computation (keyword-based)
        const sparseScores = this.computeKeywordScores(keywordMatches);
        const sparseIntent = Object.entries(sparseScores).reduce((max, [intent, score]) => 
            score > sparseScores[max] ? intent : max, 'general');
        const sparseScore = sparseScores[sparseIntent];
        
        const t3 = performance.now();

        // Hybrid Logic: Convex Scaling Fusion
        let finalIntent = denseIntent;
        let confidence = denseScore;
        let handler = "semantic-only";

        if (denseIntent === sparseIntent) {
            // Both agree - use weighted combination
            confidence = this.alpha * denseScore + (1 - this.alpha) * sparseScore;
            finalIntent = denseIntent;
            handler = "hybrid-consensus";
        } else if (keywordMatches.length > 0 && keywordMatches[0].intent === denseIntent) {
            // Keyword confirms dense intent
            confidence = 0.5 * denseScore + 0.5 * keywordMatches[0].score;
            finalIntent = denseIntent;
            handler = "keyword-dense-confirm";
        } else {
            // Disagreement - use weighted combination, choose higher scorer
            confidence = this.alpha * denseScore + (1 - this.alpha) * sparseScore;
            finalIntent = denseScore > sparseScore ? denseIntent : sparseIntent;
            handler = "weighted-choice";
        }

        // Fallback Check (threshold 0.5)
        if (confidence < 0.5) {
            logger.warn('API', 'Fallback triggered', {
                requestId,
                query: text,
                top_intent: finalIntent,
                dense_score: denseScore,
                sparse_score: sparseScore,
                final_confidence: confidence
            });
            handler = "fallback";
            finalIntent = "general";
        }

        const timing = {
            keyword_search: t1 - t0,
            embedding_compute: t2 - t1,
            similarity_compute: t3 - t2,
            total: t3 - t_start
        };

        // Map intent to adapter_id (Placeholder for Phase 3)
        const adapterMap: Record<string, string> = {
            "legal": "lora-legal-v1",
            "coding": "lora-coding-v1",
            "general": "base-model"
        };

        const response: RouterResponse = {
            intent: finalIntent,
            confidence,
            adapter_id: adapterMap[finalIntent] || "base-model",
            handler,
            timing_ms: timing
        };

        logger.info('API', 'Intent detection complete', {
            requestId,
            metadata: response
        });

        return response;
    }

    private checkKeywords(text: string): string | null {
        const lower = text.toLowerCase();
        for (const [keyword, intent] of Object.entries(this.keywordMap)) {
            if (lower.includes(keyword)) {
                return intent;
            }
        }
        return null;
    }

    private checkKeywordsMulti(text: string): Array<{ keyword: string; intent: string; score: number }> {
        const lower = text.toLowerCase();
        const matches: Array<{ keyword: string; intent: string; score: number }> = [];
        
        for (const [keyword, intent] of Object.entries(this.keywordMap)) {
            if (lower.includes(keyword)) {
                // Fixed score per keyword (simplified vs full BM25)
                matches.push({ keyword, intent, score: 0.8 });
            }
        }
        
        return matches;
    }

    private computeKeywordScores(matches: Array<{ keyword: string; intent: string; score: number }>): Record<string, number> {
        const scores: Record<string, number> = {
            legal: 0,
            coding: 0,
            general: 0
        };

        if (matches.length === 0) {
            return scores;
        }

        // Aggregate scores by intent
        for (const match of matches) {
            scores[match.intent] += match.score;
        }

        // Normalize to 0-1 range
        const maxScore = Math.max(...Object.values(scores));
        if (maxScore > 0) {
            for (const intent in scores) {
                scores[intent] = scores[intent] / maxScore;
            }
        }

        return scores;
    }

    private computeBestMatch(queryEmbedding: any): { bestIntent: string, maxScore: number } {
        let bestIntent = "general";
        let maxScore = -1;

        for (const [intent, embeddings] of Object.entries(this.prototypeEmbeddings)) {
            for (const embedding of embeddings) {
                const score = this.cosineSimilarity(queryEmbedding, embedding);
                if (score > maxScore) {
                    maxScore = score;
                    bestIntent = intent;
                }
            }
        }
        return { bestIntent, maxScore };
    }

    private cosineSimilarity(a: any, b: any): number {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
