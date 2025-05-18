const DOCUMENTS_BASE_URL = `${import.meta.env.VITE_API_BASE_URL}documents`;
const CHAT_BACKEND_URL = `${import.meta.env.VITE_API_BASE_URL}chat`;

/**
 * Fetches the currently uploaded document's name.
 * @returns {Promise<string|null>} The name of the document or null.
 * @throws {Error} If the request fails or the response is not ok.
 */
export const fetchUploadedDocumentName = async () => {
  console.log("API Service: Fetching uploaded document...");
  const res = await fetch(DOCUMENTS_BASE_URL);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `Failed to fetch documents: ${res.status}` }));
    throw new Error(errorData.detail || `Failed to fetch documents: ${res.status}`);
  }
  const data = await res.json();
  if (data.documents && data.documents.length > 0) {
    return data.documents[0].name; // Assuming only one document
  }
  return null;
};

/**
 * Uploads a new PDF document.
 * If a document already exists, it's deleted first.
 * @param {File} selectedFile - The PDF file to upload.
 * @param {string|null} currentUploadedDocumentName - The name of the currently uploaded document, if any.
 * @returns {Promise<string>} The name of the newly uploaded document.
 * @throws {Error} If the upload process fails.
 */
export const uploadDocument = async (selectedFile, currentUploadedDocumentName) => {
  console.log("API Service: Uploading document...");
  if (!selectedFile) {
    throw new Error("No file selected to upload.");
  }

  // If a document already exists, delete it first
  if (currentUploadedDocumentName) {
    console.log("API Service: Removing existing document before new upload...");
    await deleteUploadedDocument(); // Use the delete service function
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  const uploadRes = await fetch(`${DOCUMENTS_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!uploadRes.ok) {
    const errorData = await uploadRes.json().catch(() => ({ detail: `Failed to upload document: ${uploadRes.status}` }));
    throw new Error(errorData.detail || `Failed to upload document: ${uploadRes.status}`);
  }
  // Assuming the upload endpoint doesn't return the new name directly,
  // we rely on fetching it again or the calling component can do that.
  // For simplicity, let's assume the caller will re-fetch the document name.
  // const uploadData = await uploadRes.json();
  await uploadRes.json(); // Consume response body
  console.log("API Service: Document uploaded, caller should re-fetch name.");
  // To ensure the latest name is returned, we could fetch it here,
  // but it might be better to let the component orchestrate this.
  // For now, we'll just signal success. The component will call fetchUploadedDocumentName.
  return selectedFile.name; // Or fetch and return the actual name from server if API supports it
};


/**
 * Deletes the currently uploaded document.
 * @returns {Promise<void>}
 * @throws {Error} If the deletion fails.
 */
export const deleteUploadedDocument = async () => {
  console.log("API Service: Deleting document...");
  const res = await fetch(DOCUMENTS_BASE_URL, { method: "DELETE" });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `Failed to delete document: ${res.status}` }));
    throw new Error(errorData.detail || `Failed to delete document: ${res.status}`);
  }
  await res.json();
  console.log("API Service: Document deleted successfully.");
};

/**
 * Sends a chat message to the backend.
 * @param {string} message - The user's message.
 * @param {string} userId - The ID of the user.
 * @returns {Promise<string>} The AI's response.
 * @throws {Error} If the request fails or the response is not ok.
 */
export const sendChatMessage = async (message, userId = "user_123") => {
  console.log("API Service: Sending chat message...");
  const res = await fetch(CHAT_BACKEND_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, user_id: userId }),
  });

  if (!res.ok) {
    const errorText = await res.text();
    let detail = errorText;
    try {
      const jsonData = JSON.parse(errorText);
      detail = jsonData.detail || errorText;
    } catch {
      // Not JSON, use raw text
    }
    throw new Error(`HTTP error! Status: ${res.status} - ${detail}`);
  }

  const data = await res.json();
  return data.response || "Sorry, I could not get a response.";
};