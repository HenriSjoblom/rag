import React, { useState, useRef, useEffect } from "react";
import {
  fetchUploadedDocumentName,
  uploadDocument,
  deleteUploadedDocument,
} from "../services/api";

function DocumentManager({
  onDocumentNameChange, // Callback to inform parent about document name
  onProcessingStateChange, // Callback to inform parent about processing state
}) {
  const [uploadedDocName, setUploadedDocName] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [docError, setDocError] = useState(null);
  const [currentSelectedFile, setCurrentSelectedFile] = useState(null);
  const internalFileInputRef = useRef(null);

  const loadDocument = async () => {
    console.log("DocumentManager: Fetching initial uploaded document...");
    setIsProcessing(true);
    onProcessingStateChange(true);
    setDocError(null);
    try {
      const name = await fetchUploadedDocumentName();
      setUploadedDocName(name);
      onDocumentNameChange(name);
    } catch (err) {
      console.error("DocumentManager: Failed to fetch document:", err);
      setDocError(err.message || "Could not check for existing document.");
      setUploadedDocName(null);
      onDocumentNameChange(null);
    } finally {
      setIsProcessing(false);
      onProcessingStateChange(false);
    }
  };

  useEffect(() => {
    loadDocument();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // Adding onDocumentNameChange and onProcessingStateChange to deps can cause loops if not memoized by parent.

  const handleFileSelectInternal = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      setCurrentSelectedFile(file);
      setDocError(null);
    } else {
      setCurrentSelectedFile(null);
      setDocError("Please select a PDF file.");
    }
  };

  const handleFileUploadInternal = async () => {
    if (!currentSelectedFile) {
      setDocError("No file selected to upload.");
      return;
    }

    setIsProcessing(true);
    onProcessingStateChange(true);
    setDocError(null);
    try {
      await uploadDocument(currentSelectedFile, uploadedDocName); // Pass current doc name for potential deletion
      await loadDocument(); // Re-fetch to get the accurate name and update parent
      setCurrentSelectedFile(null);
      if (internalFileInputRef.current) {
        internalFileInputRef.current.value = "";
      }
    } catch (err) {
      console.error("DocumentManager: Document upload failed:", err);
      setDocError(err.message || "An error occurred during upload.");
      await loadDocument(); // Still try to load current state
    } finally {
      setIsProcessing(false);
      onProcessingStateChange(false);
    }
  };

  const handleDeleteDocumentInternal = async () => {
    if (!uploadedDocName) return;

    setIsProcessing(true);
    onProcessingStateChange(true);
    setDocError(null);
    try {
      await deleteUploadedDocument();
      setUploadedDocName(null);
      onDocumentNameChange(null);
      setCurrentSelectedFile(null);
      if (internalFileInputRef.current) {
        internalFileInputRef.current.value = "";
      }
      console.log("DocumentManager: Document deleted successfully.");
    } catch (err) {
      console.error("DocumentManager: Failed to delete document:", err);
      setDocError(err.message || "Could not delete document.");
      await loadDocument(); // Still try to load current state
    } finally {
      setIsProcessing(false);
      onProcessingStateChange(false);
    }
  };

  return (
    <div className="p-3 border-b border-gray-200 bg-gray-50 text-sm">
      {isProcessing && (
        <p className="text-blue-600 animate-pulse">Processing document...</p>
      )}
      {docError && !isProcessing && (
        <p className="text-red-600 py-1">Error: {docError}</p>
      )}

      {!isProcessing && uploadedDocName && (
        <div className="flex items-center justify-between">
          <p>
            Current document:{" "}
            <span className="font-semibold">{uploadedDocName}</span>
          </p>
          <button
            onClick={handleDeleteDocumentInternal}
            className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-xs"
            disabled={isProcessing}
          >
            Remove
          </button>
        </div>
      )}

      {!isProcessing && !uploadedDocName && (
        <p className="text-gray-600 py-1">
          No document uploaded. Upload a PDF to ask questions about it.
        </p>
      )}

      <div className="mt-2 flex items-center gap-2">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileSelectInternal}
          ref={internalFileInputRef}
          className="block w-full text-xs text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
          disabled={isProcessing}
        />
        <button
          onClick={handleFileUploadInternal}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-xs disabled:opacity-50"
          disabled={!currentSelectedFile || isProcessing}
        >
          {uploadedDocName ? "Replace" : "Upload"}
        </button>
      </div>
    </div>
  );
}

export default DocumentManager;
