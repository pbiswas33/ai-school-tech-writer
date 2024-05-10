## Project Name

This project is a repository for implementing various AI models using OpenAI.

### Workflow

The workflow `update_readme.yaml` has been updated to include an additional condition check. The workflow will now only trigger the update-readme job if the pull request is merged and the title does not contain 'AI PR'.

### Error Handling

An error occurred while attempting to make LLM. The error code 401 was received with the message: 'Incorrect API key provided: sk-U82DG***************************************5lWJ. You can find your API key at [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)'.

Please ensure to provide the correct API key to avoid this error.

### UI Implementation

- Added UI functionality to accept PDF uploads and upload them to the Vector Database.
- Implemented UI to accept prompts and provide responses from the AI model.

### Code Changes
- Updated `pyqt.py` to include UI implementation for file upload, prompt entry, and response generation.
- Updated `requirements.txt` to include new dependencies for UI implementation.
- Added `upload.py` for preparing and uploading documents to the vector database.
- Updated `utility.py` for improved code structure.

---

Feel free to further update the README as needed with additional information or instructions.