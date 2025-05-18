import React, { useState, useRef, useEffect } from "react";

function ChatInterface() {
  console.log("ChatInterface component rendered");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // New state for document management
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

      const uploadData = await uploadRes.json();
      // After successful upload, fetch the document name again to confirm
      // Or, if the upload response directly gives the name, use that.
      // For now, relying on fetchUploadedDocument to get the name from the list endpoint.
      await fetchUploadedDocument();
      setSelectedFile(null); // Clear selected file
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // Reset file input
      }
      // Optionally, display a success message from uploadData.message
      console.log("Document uploaded successfully:", uploadData.message);
    } catch (err) {
      console.error("Document upload failed:", err);
      setDocumentError(err.message || "An error occurred during upload.");
      // If upload failed after delete, try to refetch to see if old doc is truly gone
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
        const errorData = await res.text();
        throw new Error(`HTTP error! Status: ${res.status} - ${errorData}`);
      }

      const data = await res.json();
      const aiResponse = data.response || "Sorry, I could not get a response.";

      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: aiResponse },
      ]);
    } catch (err) {
      console.error("Failed to fetch:", err);
      setError(err.message || "Failed to connect to the backend.");
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: `Error: ${err.message || "Failed to connect"}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col w-full max-w-3xl mx-auto h-[calc(100vh-4rem)] bg-white shadow-lg rounded-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4 text-center shadow-md">
        <h1 className="text-xl md:text-2xl font-bold tracking-tight">
          User Manual Assistant with RAG Technology
        </h1>
        <p className="text-xs md:text-sm opacity-90">
          Ask me anything about your user manuals!
        </p>
      </div>

      {/* Document Management Section */}
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

      <div className="flex-grow overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-100">
        {" "}
        {/* Changed chat bg */}
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
            disabled={isLoading || !uploadedDocumentName} // Disable chat if no document or loading
          />
          <button
            type="submit"
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
            disabled={isLoading || !query.trim() || !uploadedDocumentName} // Also disable if no document
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
    </div>
  );
}

export default ChatInterface;
