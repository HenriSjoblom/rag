import { useState, useRef, useEffect } from "react";
import ChatHeader from "./ChatHeader";
import DocumentManager from "./DocumentManager";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";
import {
  fetchUploadedDocumentName,
  uploadDocument,
  deleteUploadedDocument,
  sendChatMessage,
} from "../services/api";

function ChatInterface() {
  console.log("ChatInterface component rendered");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const [uploadedDocumentName, setUploadedDocumentName] = useState(null);
  const [isDocumentProcessing, setIsDocumentProcessing] = useState(false);
  const [documentError, setDocumentError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const loadInitialDocument = async () => {
    console.log("ChatInterface: Fetching initial uploaded document...");
    setIsDocumentProcessing(true);
    setDocumentError(null);
    try {
      const name = await fetchUploadedDocumentName();
      setUploadedDocumentName(name);
    } catch (err) {
      console.error("ChatInterface: Failed to fetch document:", err);
      setDocumentError(err.message || "Could not check for existing document.");
      setUploadedDocumentName(null);
    } finally {
      setIsDocumentProcessing(false);
    }
  };

  useEffect(() => {
    loadInitialDocument();
  }, []);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      setSelectedFile(file);
      setDocumentError(null);
    } else {
      setSelectedFile(null);
      setDocumentError("Please select a PDF file.");
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setDocumentError("No file selected to upload.");
      return;
    }

    setIsDocumentProcessing(true);
    setDocumentError(null);
    try {
      // The uploadDocument service function now handles deleting existing doc if necessary
      await uploadDocument(selectedFile, uploadedDocumentName);
      // After successful upload, refresh the document name
      await loadInitialDocument(); // Re-fetch to get the accurate name
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      console.error("ChatInterface: Document upload failed:", err);
      setDocumentError(err.message || "An error occurred during upload.");
      // Optionally, refresh document list even on failure to reflect actual state
      await loadInitialDocument();
    } finally {
      setIsDocumentProcessing(false);
    }
  };

  const handleDeleteDocument = async () => {
    if (!uploadedDocumentName) return;

    setIsDocumentProcessing(true);
    setDocumentError(null);
    try {
      await deleteUploadedDocument();
      setUploadedDocumentName(null); // Optimistically update UI
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      console.log("ChatInterface: Document deleted successfully.");
    } catch (err) {
      console.error("ChatInterface: Failed to delete document:", err);
      setDocumentError(err.message || "Could not delete document.");
      // Optionally, refresh document list even on failure
      await loadInitialDocument();
    } finally {
      setIsDocumentProcessing(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event); // Call the existing handleSubmit which now uses the service
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const currentQuery = query.trim();
    if (!currentQuery || isLoading || !uploadedDocumentName) return; // Ensure doc is uploaded

    setError(null);
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "user", text: currentQuery },
    ]);
    setQuery("");
    setIsLoading(true);

    try {
      const aiResponse = await sendChatMessage(
        currentQuery /*, "user_123" // userId can be passed if needed */
      );
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: aiResponse },
      ]);
    } catch (err) {
      console.error("ChatInterface: Failed to send/receive chat message:", err);
      const errorMessage = err.message || "Failed to connect to the backend.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col w-full max-w-3xl mx-auto h-[calc(100vh-4rem)] bg-white shadow-lg rounded-lg overflow-hidden">
      <ChatHeader />
      <DocumentManager
        isDocumentProcessing={isDocumentProcessing}
        documentError={documentError}
        uploadedDocumentName={uploadedDocumentName}
        handleDeleteDocument={handleDeleteDocument}
        handleFileSelect={handleFileSelect}
        handleFileUpload={handleFileUpload}
        selectedFile={selectedFile}
        fileInputRef={fileInputRef}
      />
      <MessageList
        messages={messages}
        isLoading={isLoading}
        error={error}
        messagesEndRef={messagesEndRef}
      />
      <MessageInput
        query={query}
        setQuery={setQuery}
        handleKeyDown={handleKeyDown}
        handleSubmit={handleSubmit}
        uploadedDocumentName={uploadedDocumentName}
        isLoading={isLoading}
      />
    </div>
  );
}

export default ChatInterface;
