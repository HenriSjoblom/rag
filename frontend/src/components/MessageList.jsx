function MessageList({ messages, isLoading, error, messagesEndRef }) {
  return (
    <div className="flex-grow overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-100">
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
                : "bg-white text-gray-800 border border-gray-200"
            }`}
          >
            <p className="text-sm md:text-base whitespace-pre-wrap">
              {msg.text}
            </p>
          </div>
        </div>
      ))}
      {isLoading && (
        <div className="flex justify-start">
          <div className="px-4 py-2 rounded-lg bg-gray-200 text-gray-600 border border-gray-200 animate-pulse max-w-[70%]">
            <p className="text-sm md:text-base">Thinking...</p>
          </div>
        </div>
      )}
      {error && !isLoading && (
        <div className="flex justify-start">
          <div className="px-4 py-2 rounded-lg bg-red-100 text-red-700 border border-red-200 max-w-[70%]">
            <p className="text-sm md:text-base whitespace-pre-wrap">
              {error}
            </p>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;