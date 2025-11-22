import {fileURLToPath} from "url";
import path from "path";
import {getLlama, type Token} from "node-llama-cpp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({

    modelPath: path.join(process.env.HOME!, ".node-llama-cpp", "models", "hf_Qwen_Qwen3-4B.Q8_0.gguf")

});
const context = await model.createContext();
const sequence = context.getSequence();

const q1 = "Hi there, how are you?";
console.log("User: " + q1);

const tokens = model.tokenize("USER: " + q1 + "\nASSISTANT: ");
const res: Token[] = [];
for await (const generatedToken of sequence.evaluate(tokens)) {
    res.push(generatedToken);

    // It's important to not concatenate the results as strings,
    // as doing so breaks some characters (like some emojis)
    // that consist of multiple tokens.
    // By using an array of tokens, we can decode them correctly together.
    const resString = model.detokenize(res);

    const lastPart = resString.split("ASSISTANT:").pop();
    if (lastPart?.includes("USER:"))
        break;
}

const a1 = model.detokenize(res).split("USER:")[0]!;
console.log("AI: " + a1.trim());