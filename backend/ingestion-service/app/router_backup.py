# Backup of the problematic delete collection function
# This is the corrected version:


@router.delete(
    "/collection",
    status_code=status.HTTP_200_OK,
    summary="Delete all files and clear ChromaDB collection",
    description="Deletes all PDF files from the source directory and clears the ChromaDB collection.",
    tags=["ingestion"],
)
async def delete_collection(
    settings: Settings = Depends(get_settings),
):
    """
    Delete all PDF files from the source directory and clear the ChromaDB collection.
    """
    collection_name = settings.CHROMA_COLLECTION_NAME
    source_directory = Path(settings.SOURCE_DIRECTORY)
    messages = []
    deleted_files_count = 0
    files_deleted_successfully = False
    collection_deleted_successfully = False

    # Delete files from source directory
    logger.info(
        f"Attempting to delete all files from source directory: '{source_directory}'"
    )
    try:
        if not source_directory.exists():
            logger.warning(f"Source directory '{source_directory}' does not exist.")
            messages.append(f"Source directory '{source_directory}' does not exist.")
        else:
            all_files = list(source_directory.iterdir())
            for file_path in all_files:
                if file_path.is_file():  # Only delete files, not directories
                    try:
                        file_path.unlink()
                        deleted_files_count += 1
                        logger.debug(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(
                            f"Failed to delete file {file_path}: {e}", exc_info=True
                        )
                        messages.append(f"Failed to delete file {file_path}: {e}")

        if not messages:  # If no errors during file deletion
            files_deleted_successfully = True

        log_msg = f"Successfully deleted {deleted_files_count} file(s) from '{source_directory}'."
        logger.info(log_msg)
        messages.append(log_msg)

    except Exception as e:
        err_msg = (
            f"An error occurred while deleting files from '{source_directory}': {e}"
        )
        logger.error(err_msg, exc_info=True)
        messages.append(err_msg)

    # Delete ChromaDB collection
    logger.info(f"Attempting to delete ChromaDB collection: '{collection_name}'")
    try:
        client = get_chroma_client(settings)
        client.delete_collection(name=collection_name)
        collection_deleted_successfully = True
        msg = f"Successfully deleted ChromaDB collection: '{collection_name}'"
        logger.info(msg)
        messages.append(msg)

        if global_vector_store_cache is not None:
            logger.info("Resetting cached LangChain Chroma vector store instance.")
            global_vector_store_cache = None

        # Also reset the actual vector store cache in the ingestion processor
        ingestion_processor_module._vector_store = None

    except Exception as e:
        # Check if the error is about collection not existing
        error_str = str(e).lower()
        if (
            "not found" in error_str
            or "does not exist" in error_str
            or "collection" in error_str
        ):
            collection_deleted_successfully = (
                True  # Desired state (collection doesn't exist)
            )
            msg = f"Collection '{collection_name}' not found. No deletion performed."
            logger.info(msg)
            messages.append(msg)

            if global_vector_store_cache is not None:
                logger.info(
                    "Resetting cached LangChain Chroma vector store instance (collection not found)."
                )
                global_vector_store_cache = None

            # Also reset the actual vector store cache in the ingestion processor
            ingestion_processor_module._vector_store = None
        else:  # Actual error occurred
            err_msg = f"Failed to delete collection '{collection_name}': {e}"
            logger.error(err_msg, exc_info=True)
            messages.append(err_msg)
            collection_deleted_successfully = False

    # Determine overall status
    if collection_deleted_successfully and files_deleted_successfully:
        final_status_code = status.HTTP_200_OK
        final_message = "ChromaDB collection and source documents cleared successfully."
    elif (
        collection_deleted_successfully or files_deleted_successfully
    ):  # Partial success
        final_status_code = status.HTTP_207_MULTI_STATUS
        final_message = "Partial success in clearing resources. Check details."
    else:  # Both failed or had significant errors
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        final_message = "Failed to clear ChromaDB collection and/or source documents."

    return JSONResponse(
        status_code=final_status_code,
        content={
            "message": final_message,
            "details": messages,
            "files_deleted_count": deleted_files_count,
            "collection_deleted": collection_deleted_successfully,
            "source_files_cleared": files_deleted_successfully,
        },
    )
