import { getLlama, LlamaChatSession, LlamaContext, LlamaModel } from "node-llama-cpp";
import path from "path";

export class LlamaService {
    private llama: any;
    private model: LlamaModel | null = null;
    private context: LlamaContext | null = null;
    private session: LlamaChatSession | null = null;

    constructor() { }

    async initialize() {
        this.llama = await getLlama();
        const modelPath = path.join(process.env.HOME!, ".node-llama-cpp", "models", "hf_Qwen_Qwen3-4B.Q8_0.gguf");

        console.log("Loading model from:", modelPath);
        this.model = await this.llama.loadModel({
            modelPath: modelPath
        });

        this.context = await this.model!.createContext();
        this.session = new LlamaChatSession({
            contextSequence: this.context!.getSequence()
        });
        console.log("LlamaService initialized.");
    }

    async chat(message: string): Promise<string> {
        if (!this.session) {
            throw new Error("LlamaService not initialized");
        }
        const response = await this.session.prompt(message);
        return response.trim();  // Remove leading \n\n from model output
    }

    get isReady(): boolean {
        return this.session !== null;
    }
}

export const llamaService = new LlamaService();         