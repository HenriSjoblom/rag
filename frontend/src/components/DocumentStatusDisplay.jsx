function DocumentStatusDisplay({
  isParentProcessing, // Overall processing state from DocumentManager
  docError, // Error message from DocumentManager
  uploadedDocName, // Name of the currently uploaded document
  onInitiateDelete, // Callback to DocumentManager to start the delete process
}) {
  if (isParentProcessing) {
    return (
      <p className="text-blue-600 animate-pulse">Processing document...</p>
    );
  }

  if (docError) {
    return <p className="text-red-600 py-1">Error: {docError}</p>;
  }

  if (uploadedDocName) {
    return (
      <div className="flex items-center justify-between">
        <p>
          Current document:{" "}
          <span className="font-semibold">{uploadedDocName}</span>
        </p>
        <button
          onClick={onInitiateDelete}
          className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-xs"
          disabled={isParentProcessing} // Disable if parent is busy
        >
          Remove
        </button>
      </div>
    );
  }

  return (
    <p className="text-gray-600 py-1">
      No document uploaded. Upload a PDF to ask questions about it.
    </p>
  );
}

export default DocumentStatusDisplay;
