import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
        <main className="flex-grow container mx-auto p-4 flex">
             <Routes>
               <Route path="/" element={<ChatInterface />} />
             </Routes>
        </main>
    </div>
  );
}

export default App;