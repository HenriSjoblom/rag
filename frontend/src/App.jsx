import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="min-h-screen p-4 flex flex-col items-center">
        <h1 className="text-3xl font-bold text-blue-600 mb-6">RAG Chat</h1>
        <Routes>
          <Route path="/" element={<ChatInterface />} />
        </Routes>
    </div>
  );
}

export default App;