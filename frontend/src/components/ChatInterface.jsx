import React, { useState, useRef, useEffect } from "react";

function ChatInterface() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const backendUrl = "http://127.0.0.1:8001/api/v1/chat";

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Scroll to bottom whenever messages update
  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]); // Also trigger on isLoading change to scroll past loading indicator

  // Handle Enter key press in textarea (submit on Enter, new line on Shift+Enter)
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault(); // Prevent default newline behavior
      handleSubmit(event); // Trigger form submission
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const currentQuery = query.trim();
    if (!currentQuery || isLoading) return; // Don't submit empty or while loading

    setError(null);

    // Add user message to history immediately
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "user", text: currentQuery },
    ]);

    setQuery(""); // Clear input field
    setIsLoading(true); // Show loading indicator

    try {
      const res = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // Send the current query, not the state which is already cleared
        body: JSON.stringify({ message: currentQuery, user_id: "user_123" }),
      });

      if (!res.ok) {
        const errorData = await res.text(); // Try to get error details
        throw new Error(`HTTP error! Status: ${res.status} - ${errorData}`);
      }

      const data = await res.json();
      const aiResponse = data.response || "Sorry, I could not get a response.";

      // Add AI response to history
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: aiResponse },
      ]);
    } catch (err) {
      console.error("Failed to fetch:", err);
      setError(err.message || "Failed to connect to the backend.");
      // Add an error message to the chat history
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: `Error: ${err.message || "Failed to connect"}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // Main chat container: flex-col, takes full height, defines width and centers
    <div className="flex flex-col w-full max-w-3xl mx-auto h-[calc(100vh-8rem)] bg-white shadow-lg rounded-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4 text-center shadow-md">
        <h1 className="text-xl md:text-2xl font-bold tracking-tight">
          RAG Powered AI Assistant
        </h1>
        <p className="text-xs md:text-sm opacity-90">
          Ask me anything about your documents!
        </p>
      </div>
      {/* Message display area: grows, scrolls */}
      <div className="flex-grow overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-2 rounded-lg max-w-[80%] md:max-w-[70%] shadow-sm ${
                msg.sender === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-white text-gray-800 border border-gray-200" // Changed AI bubble style
              }`}
            >
              {/* Render text preserving whitespace and newlines */}
              <p className="text-sm md:text-base whitespace-pre-wrap">
                {msg.text}
              </p>
            </div>
          </div>
        ))}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="px-4 py-2 rounded-lg bg-gray-200 text-gray-600 border border-gray-200 animate-pulse max-w-[70%]">
              <p className="text-sm md:text-base">Thinking...</p>
            </div>
          </div>
        )}

        {/* Error Display (Inline) */}
        {error && !isLoading && (
          <div className="flex justify-start">
            <div className="px-4 py-2 rounded-lg bg-red-100 text-red-700 border border-red-200 max-w-[70%]">
              <p className="text-sm md:text-base whitespace-pre-wrap">
                {error}
              </p>
            </div>
          </div>
        )}

        {/* Invisible element to scroll to */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area: fixed at bottom */}
      <div className="p-3 md:p-4 border-t border-gray-200 bg-white">
        <form
          onSubmit={handleSubmit}
          className="flex items-center gap-2 md:gap-3"
        >
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message..."
            className="flex-grow p-2 pr-10 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 text-sm md:text-base" // Added pr-10 for button space if overlapping
            rows="1"
            style={{ minHeight: "40px", maxHeight: "150px" }}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0" // Added flex-shrink-0
            disabled={isLoading}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-5 h-5"
            >
              <path
                d="M3.105 3.105a1.5 1.5 0 011.995-.24l13 6.5a1.5 1.5 0 010 2.79l-13 6.5a1.5 1.5 0 01-1.995-.24 1.5 1.5 0 01-.18-2.02L5.96 12 2.925 5.125a1.5 1.5 0 01.18-2.02z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatInterface;
