import express from 'express';
import { Server } from 'socket.io';
import { createServer } from 'http';
import path from 'path';
import { LlamaService } from './LlamaService.js';

import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer);

const llamaService = new LlamaService();

// Initialize LlamaService (Async)
(async () => {
  try {
    await llamaService.initialize();
    console.log("LlamaService ready to serve.");
  } catch (e) {
    console.error("Failed to initialize LlamaService:", e);
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
  console.log('A user connected');

  socket.on('disconnect', () => {
    console.log('A user disconnected');
  });

  socket.on('message', async (message: string) => {
    console.log('Received message:', message);
    
    // Echo user message back
    socket.emit('message', { sender: 'user', text: message });
    
    // Notify client that AI is thinking
    socket.emit('typing', true);
    
    try {
      const response = await llamaService.chat(message);
      socket.emit('message', { sender: 'ai', text: response });
    } catch (e) {
      console.error("Error generating response:", e);
      socket.emit('message', { sender: 'ai', text: 'Error: Could not generate response.' });
    } finally {
      socket.emit('typing', false);
    }
  });
});

httpServer.listen(8000, () => {
  console.log('Server started on port 8000');
});