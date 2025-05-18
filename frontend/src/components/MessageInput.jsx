function MessageInput({
  query,
  setQuery,
  handleKeyDown,
  handleSubmit,
  uploadedDocumentName,
  isLoading,
}) {
  return (
    <div className="p-3 md:p-4 border-t border-gray-200 bg-white">
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 md:gap-3"
      >
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            uploadedDocumentName
              ? `Ask about ${uploadedDocumentName}...`
              : "Upload a PDF to ask questions..."
          }
          className="flex-grow p-2 pr-10 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 text-sm md:text-base"
          rows="1"
          style={{ minHeight: "40px", maxHeight: "150px" }}
          disabled={isLoading || !uploadedDocumentName}
        />
        <button
          type="submit"
          className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
          disabled={isLoading || !query.trim() || !uploadedDocumentName}
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
  );
}

export default MessageInput;