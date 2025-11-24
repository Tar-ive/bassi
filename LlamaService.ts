import { getLlama, LlamaChatSession, LlamaContext, LlamaModel } from "node-llama-cpp";
import path from "path";
import { logger } from './Logger.js';
import { performance } from 'perf_hooks';

export class LlamaService {
    private llama: any;
    private model: LlamaModel | null = null;
    private context: LlamaContext | null = null;
    private session: LlamaChatSession | null = null;

    constructor() { }

    async initialize() {
        this.llama = await getLlama();
        const modelPath = path.join(process.env.HOME!, ".node-llama-cpp", "models", "hf_Qwen_Qwen3-4B.Q8_0.gguf");

        logger.info('SYS', 'Loading model', { modelPath });
        this.model = await this.llama.loadModel({
            modelPath: modelPath
        });

        this.context = await this.model!.createContext();
        this.session = new LlamaChatSession({
            contextSequence: this.context!.getSequence()
        });
        logger.info('SYS', 'LlamaService initialized');
    }

    async chat(message: string, requestId?: string): Promise<string> {
        if (!this.session) {
            throw new Error("LlamaService not initialized");
        }
        
        // Log input to model
        logger.info('AI', 'Sending prompt to model', {
            requestId,
            input_length: message.length,
            input_preview: message.substring(0, 100)
        });
        
        const t0 = performance.now();
        const response = await this.session.prompt(message);
        const t1 = performance.now();
        
        // Log output from model
        logger.info('AI', 'Model response received', {
            requestId,
            output_length: response.length,
            inference_time_ms: t1 - t0
        });
        
        return response.trim();  // Remove leading \n\n from model output
    }

    get isReady(): boolean {
        return this.session !== null;
    }
}

export const llamaService = new LlamaService();         