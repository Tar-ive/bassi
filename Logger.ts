import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

export type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';
export type LogCategory = 'APP' | 'SYS' | 'AI' | 'API';

export interface LogEntry {
    timestamp: string;
    level: LogLevel;
    category: LogCategory;
    requestId?: string;
    sessionId?: string;
    message: string;
    metadata?: Record<string, any>;
}

export class Logger {
    private static instance: Logger;
    private logFilePath: string;

    private constructor() {
        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        this.logFilePath = path.join(__dirname, 'logs.json');
    }

    public static getInstance(): Logger {
        if (!Logger.instance) {
            Logger.instance = new Logger();
        }
        return Logger.instance;
    }

    public log(level: LogLevel, category: LogCategory, message: string, meta?: { requestId?: string, sessionId?: string, [key: string]: any }) {
        const { requestId, sessionId, ...restMeta } = meta || {};

        const entry: LogEntry = {
            timestamp: new Date().toISOString(),
            level,
            category,
            requestId,
            sessionId,
            message,
            metadata: Object.keys(restMeta).length > 0 ? restMeta : undefined
        };

        const jsonLine = JSON.stringify(entry);
        
        // Console output
        console.log(jsonLine);
        
        // File output (append with newline)
        fs.appendFileSync(this.logFilePath, jsonLine + '\n');
    }

    public info(category: LogCategory, message: string, meta?: any) {
        this.log('INFO', category, message, meta);
    }

    public warn(category: LogCategory, message: string, meta?: any) {
        this.log('WARN', category, message, meta);
    }

    public error(category: LogCategory, message: string, meta?: any) {
        this.log('ERROR', category, message, meta);
    }

    public debug(category: LogCategory, message: string, meta?: any) {
        this.log('DEBUG', category, message, meta);
    }

    // Special handler for AI responses to parse thinking tags
    public logAIResponse(requestId: string, sessionId: string, fullResponse: string) {
        const thinkingRegex = /<thinking>(.*?)<\/thinking>/gs;
        const match = thinkingRegex.exec(fullResponse);

        let thinkingContent = null;
        let cleanResponse = fullResponse;

        if (match) {
            thinkingContent = match[1].trim();
            cleanResponse = fullResponse.replace(thinkingRegex, '').trim();

            this.info('AI', 'Model thinking process', {
                requestId,
                sessionId,
                thinking_content: thinkingContent,
                has_thinking: true
            });
        }

        this.info('AI', 'Model response generated', {
            requestId,
            sessionId,
            response_length: cleanResponse.length,
            has_thinking: !!thinkingContent
        });

        return { thinkingContent, cleanResponse };
    }
}

export const logger = Logger.getInstance();
