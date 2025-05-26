const DOCUMENTS_BASE_URL = `${import.meta.env.VITE_API_BASE_URL}documents`;
const CHAT_BACKEND_URL = `${import.meta.env.VITE_API_BASE_URL}chat`;

/**
 * Fetches the currently uploaded document's name.
 * @returns {Promise<string|null>} The name of the document or null.
 * @throws {Error} If the request fails or the response is not ok.
 */
export const fetchUploadedDocumentName = async () => {
  console.log("API Service: Fetching uploaded document...");
  console.log(`Document URL: ${DOCUMENTS_BASE_URL}`);
  console.log("API Service: Fetching document name...");
  const res = await fetch(DOCUMENTS_BASE_URL);
  if (!res.ok) {
    const errorData = await res
      .json()
      .catch(() => ({ detail: `Failed to fetch documents: ${res.status}` }));
    throw new Error(
      errorData.detail || `Failed to fetch documents: ${res.status}`
    );
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
 * @returns {Promise<Object>} The upload response
 * @throws {Error} If the upload process fails.
 */
export const uploadDocument = async (
  selectedFile,
  currentUploadedDocumentName
) => {
  console.log("API Service: Uploading document...");
  if (!selectedFile) {
    throw new Error("No file selected to upload.");
  }

  // If a document already exists, delete it first
  if (currentUploadedDocumentName) {
    console.log("API Service: Removing existing document before new upload...");
    await deleteUploadedDocument();
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  // Create AbortController for timeout control
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes timeout

  try {
    const uploadRes = await fetch(`${DOCUMENTS_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!uploadRes.ok) {
      let errorMessage = `Failed to upload document: ${uploadRes.status}`;

      try {
        const errorData = await uploadRes.json();
        errorMessage = errorData.detail || errorMessage;

        // Handle specific error cases
        if (uploadRes.status === 503) {
          errorMessage =
            "Backend services are currently unavailable. Please check:\n" +
            "1. RAG service is running on port 8001\n" +
            "2. Ingestion service is running on port 8002\n" +
            "3. Services can communicate with each other\n" +
            "Error: " +
            (errorData.detail || "Service unavailable");
        } else if (uploadRes.status === 409) {
          errorMessage =
            "An ingestion process is already running. Please wait and try again.";
        }
      } catch {
        // If we can't parse JSON, use status-based messages
        if (uploadRes.status === 503) {
          errorMessage =
            "Backend service is unavailable. Please check if all services are running and try again.";
        }
      }

      throw new Error(errorMessage);
    }

    // Try to parse JSON response, but don't fail if it's not JSON
    try {
      const responseData = await uploadRes.json();
      console.log("API Service: Upload response:", responseData);
      return responseData;
    } catch {
      console.log("API Service: Upload successful, but response was not JSON");
      return {
        status: "Upload accepted",
        message: "File uploaded successfully",
      };
    }
  } catch (error) {
    clearTimeout(timeoutId);

    if (error.name === "AbortError") {
      throw new Error(
        "Upload timed out after 2 minutes. The file may be too large or the service is not responding."
      );
    }

    throw error;
  }
};

/**
 * Deletes the currently uploaded document.
 * @returns {Promise<void>}
 * @throws {Error} If the deletion fails.
 */
export const deleteUploadedDocument = async () => {
  console.log("API Service: Deleting document...");
  console.log(`Document URL: ${DOCUMENTS_BASE_URL}`);
  const res = await fetch(DOCUMENTS_BASE_URL, { method: "DELETE" });
  if (!res.ok) {
    const errorData = await res
      .json()
      .catch(() => ({ detail: `Failed to delete document: ${res.status}` }));
    throw new Error(
      errorData.detail || `Failed to delete document: ${res.status}`
    );
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

/**
 * Gets the current ingestion status with retry logic
 * @param {number} retries - Number of retries
 * @returns {Promise<Object>} The ingestion status
 * @throws {Error} If the request fails
 */
export const getIngestionStatus = async (retries = 3) => {
  console.log("API Service: Getting ingestion status...");

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const res = await fetch(`${DOCUMENTS_BASE_URL}/status`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        let errorMessage = `Failed to get status: ${res.status}`;

        try {
          const errorData = await res.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          if (res.status === 503) {
            errorMessage = "Status service is unavailable";
          }
        }

        throw new Error(errorMessage);
      }

      return await res.json();
    } catch (error) {
      console.warn(
        `Status check attempt ${attempt + 1} failed:`,
        error.message
      );

      if (attempt === retries - 1) {
        throw error; // Last attempt, throw the error
      }

      // Wait before retrying
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
};

/**
 * Polls for ingestion completion with progress callbacks
 * @param {number} maxAttempts - Maximum number of polling attempts
 * @param {number} interval - Polling interval in milliseconds
 * @param {Function} onProgress - Callback for progress updates
 * @returns {Promise<Object>} Final status when completed
 */
export const waitForIngestionCompletion = async (
  maxAttempts = 60,
  interval = 2000,
  onProgress = null
) => {
  console.log("API Service: Waiting for ingestion completion...");

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const status = await getIngestionStatus();

      // Call progress callback if provided
      if (onProgress) {
        onProgress({
          attempt: attempt + 1,
          maxAttempts,
          status: status.status,
          isProcessing: status.is_processing,
          errors: status.errors,
        });
      }

      if (!status.is_processing) {
        console.log("API Service: Ingestion completed:", status);
        return status;
      }

      console.log(
        `API Service: Ingestion still processing... (attempt ${attempt + 1})`
      );
      await new Promise((resolve) => setTimeout(resolve, interval));
    } catch (error) {
      console.error("Error checking ingestion status:", error);

      // Call error callback if provided
      if (onProgress) {
        onProgress({
          attempt: attempt + 1,
          maxAttempts,
          error: error.message,
          isProcessing: true, // Assume still processing on error
        });
      }

      // Continue polling even if status check fails
      await new Promise((resolve) => setTimeout(resolve, interval));
    }
  }

  throw new Error("Ingestion polling timeout - process may still be running");
};

/**
 * Uploads a document and returns immediately with polling capability
 * @param {File} selectedFile - The PDF file to upload
 * @param {string|null} currentUploadedDocumentName - Current document name
 * @returns {Promise<Object>} Upload response
 */
export const uploadDocumentOnly = async (
  selectedFile,
  currentUploadedDocumentName
) => {
  console.log("API Service: Uploading document (no wait)...");
  return await uploadDocument(selectedFile, currentUploadedDocumentName);
};
