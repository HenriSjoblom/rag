import React, { useState, useRef, useEffect } from "react";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";
import { sendChatMessage } from "../services/api";

function ChatView({ uploadedDocumentName, isDocumentProcessing }) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false); // For chat messages
  const [error, setError] = useState(null); // For chat messages
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const currentQuery = query.trim();
    // Ensure document is uploaded and not currently processing for chat submission
    if (
      !currentQuery ||
      isLoading ||
      !uploadedDocumentName ||
      isDocumentProcessing
    ) {
      return;
    }

    setError(null);
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "user", text: currentQuery },
    ]);
    setQuery("");
    setIsLoading(true); // For chat message loading

    try {
      const aiResponse = await sendChatMessage(currentQuery);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: aiResponse },
      ]);
    } catch (err) {
      console.error("ChatView: Failed to send/receive chat message:", err);

      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: "Error: Internal server error." },
      ]);
    } finally {
      setIsLoading(false); // For chat message loading
    }
  };

  return (
    <>
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
        // isLoading for both chat loading and document processing
        isLoading={isLoading || isDocumentProcessing}
      />
    </>
  );
}

export default ChatView;
