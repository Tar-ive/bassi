import { io } from 'socket.io-client';

const socket = io('http://localhost:8000');

socket.on('connect', () => {
    console.log('Connected to server');

    // Send a test message that should trigger Legal intent
    const testMessage = "Draft a motion to dismiss for lack of jurisdiction";
    console.log(`Sending: "${testMessage}"`);
    socket.emit('message', testMessage);
});

socket.on('message', (data) => {
    if (data.sender === 'ai') {
        console.log('\n--- AI Response Received ---');
        console.log('Text:', data.text);
        if (data.metadata) {
            console.log('Metadata:', JSON.stringify(data.metadata, null, 2));
        }
        socket.disconnect();
    }
});

socket.on('disconnect', () => {
    console.log('Disconnected');
});
