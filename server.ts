import express from 'express';
import { Server } from 'socket.io';
import { createServer } from 'http';
import path from 'path';
import { LlamaService } from './LlamaService.js';
import { IntentRouter } from './IntentRouter.js';
import { logger } from './Logger.js';
import { v4 as uuidv4 } from 'uuid';

import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer);

const llamaService = new LlamaService();
const intentRouter = new IntentRouter();

// Initialize Services (Async)
(async () => {
  try {
    logger.info('SYS', 'Starting services...');
    await Promise.all([
      llamaService.initialize(),
      intentRouter.initialize()
    ]);
    logger.info('SYS', 'All Services Ready.');
  } catch (e) {
    logger.error('SYS', 'Failed to initialize services', { error: e });
  }
})();

app.use(express.static(__dirname));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/health', (req, res) => {
  if (llamaService.isReady) {
    res.status(200).json({ status: 'ok', model: 'loaded' });
  } else {
    res.status(503).json({ status: 'initializing', model: 'loading' });
  }
});

io.on('connection', (socket) => {
  const sessionId = uuidv4();
  socket.data.sessionId = sessionId;

  logger.info('SYS', 'User connected', { sessionId });

  socket.on('disconnect', () => {
    logger.info('SYS', 'User disconnected', { sessionId });
  });

  socket.on('message', async (message: string) => {
    const requestId = uuidv4();
    const t_start = performance.now(); // Start E2E timer

    logger.info('APP', 'Received message', { requestId, sessionId, message });

    // Echo user message back
    socket.emit('message', { sender: 'user', text: message });

    // Notify client that AI is thinking
    socket.emit('typing', true);

    try {
      // Phase 2: Intent Detection
      const routeResult = await intentRouter.route(message, requestId);

      // Phase 3 Placeholder: Adapter Swapping would happen here based on routeResult.adapter_id

      // Inference
      const rawResponse = await llamaService.chat(message, requestId);

      const t_end = performance.now(); // Stop E2E timer
      const e2e_latency_ms = t_end - t_start;

      // Parse Thinking Tags & Log
      const { thinkingContent, cleanResponse } = logger.logAIResponse(requestId, sessionId, rawResponse);

      // Log E2E Performance
      logger.info('API', 'Request completed', {
        requestId,
        sessionId,
        timing_ms: {
          e2e_latency: e2e_latency_ms,
          intent_detection: routeResult.timing_ms.total
        }
      });

      socket.emit('message', {
        sender: 'ai',
        text: cleanResponse,
        metadata: {
          intent: routeResult.intent,
          confidence: routeResult.confidence,
          adapter_id: routeResult.adapter_id,
          thinking: thinkingContent,
          latency_ms: e2e_latency_ms.toFixed(0)
        }
      });
    } catch (e) {
      logger.error('API', 'Error generating response', { requestId, sessionId, error: e });
      socket.emit('message', { sender: 'ai', text: 'Error: Could not generate response.' });
    } finally {
      // CRITICAL: Always turn off typing indicator to prevent UI from getting stuck
      socket.emit('typing', false);
    }
  });
});

httpServer.listen(8000, () => {
  logger.info('SYS', 'Server started on port 8000');
});