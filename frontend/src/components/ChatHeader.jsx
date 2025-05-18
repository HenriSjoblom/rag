function ChatHeader() {
  return (
    <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4 text-center shadow-md">
      <h1 className="text-xl md:text-2xl font-bold tracking-tight">
        User Manual Assistant with RAG Technology
      </h1>
      <p className="text-xs md:text-sm opacity-90">
        Ask me anything about your user manuals!
      </p>
    </div>
  );
}

export default ChatHeader;