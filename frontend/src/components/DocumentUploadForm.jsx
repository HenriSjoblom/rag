import { useState } from "react";

function DocumentUploadForm({
  uploadedDocName,
  isParentProcessing, // Indicates if DocumentManager is busy
  onInitiateUpload, // Callback to DocumentManager to start the upload process
  internalFileInputRef,
}) {
  const [currentSelectedFile, setCurrentSelectedFile] = useState(null);
  const [selectionError, setSelectionError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      setCurrentSelectedFile(file);
      setSelectionError(null);
    } else {
      setCurrentSelectedFile(null);
      setSelectionError("Please select a PDF file.");
    }
  };

  const handleUploadClick = () => {
    if (!currentSelectedFile) {
      setSelectionError("No file selected to upload.");
      return;
    }
    if (onInitiateUpload) {
      onInitiateUpload(currentSelectedFile); // Pass the selected file to the parent
    }
  };

  return (
    <>
      {selectionError && (
        <p className="text-red-500 text-xs mt-1">{selectionError}</p>
      )}
      <div className="mt-2 flex items-center gap-2">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          ref={internalFileInputRef}
          className="block w-full text-xs text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
          disabled={isParentProcessing}
        />
        <button
          onClick={handleUploadClick}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-xs disabled:opacity-50"
          disabled={!currentSelectedFile || isParentProcessing}
        >
          {uploadedDocName ? "Replace" : "Upload"}
        </button>
      </div>
    </>
  );
}

export default DocumentUploadForm;
