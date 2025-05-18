import { useState, useRef, useEffect } from "react";
import ChatHeader from "./ChatHeader";
import DocumentManager from "./DocumentManager";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

function ChatInterface() {
  console.log("ChatInterface component rendered");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const [uploadedDocumentName, setUploadedDocumentName] = useState(null);
  const [isDocumentProcessing, setIsDocumentProcessing] = useState(false); // For upload/delete loading
  const [documentError, setDocumentError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const chatBackendUrl = "http://127.0.0.1:8001/api/v1/chat";
  const documentsBaseUrl = "http://127.0.0.1:8001/api/v1/documents";

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Fetch current document on mount
  useEffect(() => {
    console.log("Component mounted, fetching uploaded document...");
    fetchUploadedDocument();
  }, []);

  const fetchUploadedDocument = async () => {
    console.log("Fetching uploaded document...");
    setIsDocumentProcessing(true);
    setDocumentError(null);
    try {
      const res = await fetch(documentsBaseUrl);
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(
          errorData.detail || `Failed to fetch documents: ${res.status}`
        );
      }
      const data = await res.json();
      if (data.documents && data.documents.length > 0) {
        setUploadedDocumentName(data.documents[0].name); // Assuming only one document as per requirement
      } else {
        setUploadedDocumentName(null);
      }
    } catch (err) {
      console.error("Failed to fetch document:", err);
      setDocumentError(err.message || "Could not check for existing document.");
      setUploadedDocumentName(null);
    } finally {
      setIsDocumentProcessing(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      setSelectedFile(file);
      setDocumentError(null); // Clear previous error
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
      // If a document already exists, delete it first to maintain the "only one document" rule
      if (uploadedDocumentName) {
        const deleteRes = await fetch(documentsBaseUrl, { method: "DELETE" });
        if (!deleteRes.ok) {
          const errorData = await deleteRes.json();
          throw new Error(
            errorData.detail ||
              `Failed to remove existing document: ${deleteRes.status}`
          );
        }
        setUploadedDocumentName(null); // Optimistically update
        console.log("Existing document removed before new upload.");
      }

      const formData = new FormData();
      formData.append("file", selectedFile);

      const uploadRes = await fetch(`${documentsBaseUrl}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const errorData = await uploadRes.json();
        throw new Error(
          errorData.detail || `Failed to upload document: ${uploadRes.status}`
        );
      }

      // const uploadData = await uploadRes.json(); // Keep if you need success message
      await uploadRes.json(); // Consume the response body
      await fetchUploadedDocument();
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      // console.log("Document uploaded successfully:", uploadData.message);
    } catch (err) {
      console.error("Document upload failed:", err);
      setDocumentError(err.message || "An error occurred during upload.");
      await fetchUploadedDocument();
    } finally {
      setIsDocumentProcessing(false);
    }
  };

  const handleDeleteDocument = async () => {
    if (!uploadedDocumentName) return;

    setIsDocumentProcessing(true);
    setDocumentError(null);
    try {
      const res = await fetch(documentsBaseUrl, { method: "DELETE" });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(
          errorData.detail || `Failed to delete document: ${res.status}`
        );
      }
      setUploadedDocumentName(null);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      console.log("Document deleted successfully.");
    } catch (err) {
      console.error("Failed to delete document:", err);
      setDocumentError(err.message || "Could not delete document.");
    } finally {
      setIsDocumentProcessing(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const currentQuery = query.trim();
    if (!currentQuery || isLoading) return;

    setError(null);

    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "user", text: currentQuery },
    ]);

    setQuery("");
    setIsLoading(true);

    try {
      const res = await fetch(chatBackendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: currentQuery, user_id: "user_123" }),
      });

      if (!res.ok) {
        const errorData = await res.text(); // Use .text() for more robust error handling
        let detail = errorData;
        try {
          const jsonData = JSON.parse(errorData); // Try to parse as JSON
          detail = jsonData.detail || errorData;
        } catch (e) {
          // If not JSON, use the raw text
        }
        throw new Error(`HTTP error! Status: ${res.status} - ${detail}`);
      }

      const data = await res.json();
      const aiResponse = data.response || "Sorry, I could not get a response.";

      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: aiResponse },
      ]);
    } catch (err) {
      console.error("Failed to fetch:", err);
      const errorMessage = err.message || "Failed to connect to the backend.";
      setError(errorMessage); // Set error state to display in MessageList
  
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
        error={error} /* Pass the chat error here */
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
