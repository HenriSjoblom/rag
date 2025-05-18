function DocumentManager({
  isDocumentProcessing,
  documentError,
  uploadedDocumentName,
  handleDeleteDocument,
  handleFileSelect,
  handleFileUpload,
  selectedFile,
  fileInputRef,
}) {
  return (
    <div className="p-3 border-b border-gray-200 bg-gray-50 text-sm">
      {isDocumentProcessing && (
        <p className="text-blue-600 animate-pulse">Processing document...</p>
      )}
      {documentError && (
        <p className="text-red-600 py-1">Error: {documentError}</p>
      )}

      {!isDocumentProcessing && uploadedDocumentName && (
        <div className="flex items-center justify-between">
          <p>
            Current document:{" "}
            <span className="font-semibold">{uploadedDocumentName}</span>
          </p>
          <button
            onClick={handleDeleteDocument}
            className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-xs"
            disabled={isDocumentProcessing}
          >
            Remove
          </button>
        </div>
      )}

      {!isDocumentProcessing && !uploadedDocumentName && (
        <p className="text-gray-600 py-1">
          No document uploaded. Upload a PDF to ask questions about it.
        </p>
      )}

      <div className="mt-2 flex items-center gap-2">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          ref={fileInputRef}
          className="block w-full text-xs text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
          disabled={isDocumentProcessing}
        />
        <button
          onClick={handleFileUpload}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-xs disabled:opacity-50"
          disabled={!selectedFile || isDocumentProcessing}
        >
          {uploadedDocumentName ? "Replace" : "Upload"}
        </button>
      </div>
    </div>
  );
}

export default DocumentManager;
