import { useState, useRef, useEffect } from "react";
import {
  fetchUploadedDocumentName,
  uploadDocument,
  deleteUploadedDocument,
} from "../services/api";
import DocumentStatusDisplay from "./DocumentStatusDisplay";
import DocumentUploadForm from "./DocumentUploadForm";

function DocumentManager({ onDocumentNameChange, onProcessingStateChange }) {
  const [uploadedDocName, setUploadedDocName] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false); // Overall processing state for this manager
  const [docError, setDocError] = useState(null);
  const internalFileInputRef = useRef(null);

  const loadDocument = async () => {
    console.log("DocumentManager: Fetching document status...");
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleInitiateUpload = async (fileToUpload) => {
    if (!fileToUpload) return;

    setIsProcessing(true);
    onProcessingStateChange(true);
    setDocError(null);
    try {
      await uploadDocument(fileToUpload, uploadedDocName);
      await loadDocument(); // Refresh document status
      // Clear the file input in DocumentUploadForm
      if (internalFileInputRef.current) {
        internalFileInputRef.current.value = "";
      }
    } catch (err) {
      console.error("DocumentManager: Document upload failed:", err);
      setDocError(err.message || "An error occurred during upload.");
      // Even on error, try to load the current state, as a partial operation might have occurred
      await loadDocument();
    }
  };

  const handleInitiateDelete = async () => {
    if (!uploadedDocName) return;

    setIsProcessing(true);
    onProcessingStateChange(true);
    setDocError(null);
    try {
      await deleteUploadedDocument();
      await loadDocument(); // Refresh document status
      console.log("DocumentManager: Document deleted successfully.");
    } catch (err) {
      console.error("DocumentManager: Failed to delete document:", err);
      setDocError(err.message || "Could not delete document.");
      await loadDocument(); // Refresh document status even on error
    }
  };

  return (
    <div className="p-3 border-b border-gray-200 bg-gray-50 text-sm">
      <DocumentStatusDisplay
        isParentProcessing={isProcessing}
        docError={docError}
        uploadedDocName={uploadedDocName}
        onInitiateDelete={handleInitiateDelete}
      />
      <DocumentUploadForm
        uploadedDocName={uploadedDocName}
        isParentProcessing={isProcessing}
        onInitiateUpload={handleInitiateUpload}
        internalFileInputRef={internalFileInputRef}
      />
    </div>
  );
}

export default DocumentManager;
