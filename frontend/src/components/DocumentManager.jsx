import { useState, useRef, useEffect } from "react";
import {
  fetchUploadedDocumentName,
  uploadDocumentOnly,
  deleteUploadedDocument,
  waitForIngestionCompletion,
} from "../services/api";
import DocumentStatusDisplay from "./DocumentStatusDisplay";
import DocumentUploadForm from "./DocumentUploadForm";

function DocumentManager({ onDocumentNameChange, onProcessingStateChange }) {
  const [uploadedDocName, setUploadedDocName] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [docError, setDocError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null); // New state for processing details
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
    setProcessingStatus({ status: "Starting upload..." });

    try {
      // First upload the file
      console.log("DocumentManager: Starting file upload...");
      await uploadDocumentOnly(fileToUpload, uploadedDocName);
      console.log(
        "DocumentManager: File uploaded successfully, waiting for ingestion..."
      );

      setProcessingStatus({ status: "File uploaded, starting ingestion..." });

      // Then wait for ingestion to complete with progress updates
      const finalStatus = await waitForIngestionCompletion(
        60, // max attempts
        2000, // interval
        (progress) => {
          // Update processing status with progress info
          setProcessingStatus({
            status: progress.status || "Processing...",
            attempt: progress.attempt,
            maxAttempts: progress.maxAttempts,
            documentsProcessed: progress.documentsProcessed,
            chunksAdded: progress.chunksAdded,
            error: progress.error,
          });
        }
      );

      if (finalStatus.errors && finalStatus.errors.length > 0) {
        console.warn(
          "DocumentManager: Ingestion completed with errors:",
          finalStatus.errors
        );
        setDocError(
          `Upload completed with warnings: ${finalStatus.errors.join(", ")}`
        );
      } else {
        console.log("DocumentManager: Ingestion completed successfully!");
        setProcessingStatus({
          status: "Completed successfully!",
          documentsProcessed: finalStatus.documents_processed,
          chunksAdded: finalStatus.chunks_added,
        });

        // Show completion status briefly before clearing
        setTimeout(() => {
          setProcessingStatus(null);
        }, 2000);
      }

      // Refresh document status
      await loadDocument();

      // Clear the file input
      if (internalFileInputRef.current) {
        internalFileInputRef.current.value = "";
      }
    } catch (err) {
      console.error("DocumentManager: Upload/ingestion failed:", err);
      setDocError(err.message || "An error occurred during upload/ingestion.");
      setProcessingStatus(null);
      // Try to refresh status even on error
      await loadDocument();
    } finally {
      setIsProcessing(false);
      onProcessingStateChange(false);
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
        processingStatus={processingStatus}
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
