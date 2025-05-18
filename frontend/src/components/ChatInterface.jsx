import { useState, useCallback } from "react";
import ChatHeader from "./ChatHeader";
import DocumentManager from "./DocumentManager";
import ChatView from "./ChatView";

function ChatInterface() {
  console.log("ChatInterface component rendered");

  // State related to document status, updated by DocumentManager
  const [currentUploadedDocName, setCurrentUploadedDocName] = useState(null);
  const [isDocCurrentlyProcessing, setIsDocCurrentlyProcessing] =
    useState(false);

  // Callbacks for DocumentManager to update ChatInterface state
  const handleDocumentNameChange = useCallback((name) => {
    setCurrentUploadedDocName(name);
  }, []);

  const handleProcessingStateChange = useCallback((isProcessing) => {
    setIsDocCurrentlyProcessing(isProcessing);
  }, []);

  return (
    <div className="flex flex-col w-full max-w-3xl mx-auto h-[calc(100vh-4rem)] bg-white shadow-lg rounded-lg overflow-hidden">
      <ChatHeader />
      <DocumentManager
        onDocumentNameChange={handleDocumentNameChange}
        onProcessingStateChange={handleProcessingStateChange}
      />
      <ChatView
        uploadedDocumentName={currentUploadedDocName}
        isDocumentProcessing={isDocCurrentlyProcessing}
      />
    </div>
  );
}

export default ChatInterface;
