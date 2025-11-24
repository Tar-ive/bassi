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

    // Keyword Scenarios from Plan
    private keywordMap: Record<string, string> = {
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

        // Layer 1: Keyword Search
        const t0 = performance.now();
        const keywordIntent = this.checkKeywords(text);
        const t1 = performance.now();

        // Layer 2: Embedding Computation
        const output = await this.classifier(text, { pooling: 'mean', normalize: true });
        const queryEmbedding = output.data;
        const t2 = performance.now();

        // Layer 3: Similarity Computation
        const { bestIntent: vectorIntent, maxScore: vectorScore } = this.computeBestMatch(queryEmbedding);
        const t3 = performance.now();

        // Hybrid Logic Decision
        let finalIntent = vectorIntent;
        let confidence = vectorScore;
        let handler = "semantic-only";

        if (keywordIntent) {
            if (keywordIntent === vectorIntent && vectorScore > 0.6) {
                confidence = 1.0;
                handler = "keyword-confirmed";
            } else if (vectorScore < 0.4) {
                // False positive keyword, trust vector (even if low) or fallback
                confidence = vectorScore;
                handler = "semantic-only";
            } else {
                // Ambiguous - default to vector
                confidence = vectorScore;
                handler = "semantic-only";
            }
        }

        // Fallback Check
        if (confidence < 0.7 && handler !== "keyword-confirmed") {
            logger.warn('API', 'Fallback triggered', {
                requestId,
                query: text,
                top_intent: finalIntent,
                top_score: confidence
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
