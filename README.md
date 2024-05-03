## Error making LLM: 1 validation error for Generation
text
  str type expected (type=type_error.str)

Fix for - improper assignment

The code changes in this commit fix an issue with improper assignment in the `utility.py` file. The `call_openai` function now correctly uses the response from the AI client for parsing instead of the input messages.