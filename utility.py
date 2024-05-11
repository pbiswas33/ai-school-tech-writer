import os
import base64
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def format_data_for_openai(diffs, readme_content, commit_messages):
    # Initialize the prompt variable to None
    prompt = None

    # Combine the changes from diffs into a single string with each change clearly delineated by filename and patch content
    changes = "\n".join(
        f'File: {diff["filename"]}\n{diff["patch"]}\n' for diff in diffs
    )

    # Concatenate all commit messages into a single string separated by new lines
    commit_messages = "\n".join(commit_messages)

    # Decode the README content from base64 to UTF-8 format
    readme_content = base64.b64decode(readme_content.content).decode("utf-8")

    # Construct the prompt with clear instructions for the language model, including code changes, commit messages, and current README content
    prompt = (
        "Please review the following code changes and commit messages from a GitHub pull request:\n"
        "Code changes from Pull Request:\n"
        f"{changes}\n"
        "Commit messages:\n"
        f"{commit_messages}"
        "Here is the current README file content:\n"
        f"{readme_content}\n"
        "Consider the code changes and commit messages, determine if the README needs to be updated. If so, edit the README, ensuring to maintain its existing style and clarity.\n"
        "Updated README:\n"
    )

    # Return the constructed prompt to be used by the OpenAI model
    return prompt

def call_openai(prompt):
    # Initialize the ChatOpenAI client with the API key and model specification
    client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125" )
    
    try:
        # Define the roles and content for the system and user in the conversation with the AI
        messages = [
            {"role": "system", "content": "You are an AI trained to help updating READMEs based on code changes and commit messages."},
            {"role": "user", "content": prompt}
        ]
        # Invoke the OpenAI client with the constructed messages and parse the response
        llm_response = client.invoke(input=messages)
        parser = StrOutputParser()
        return parser.invoke(input=llm_response)
    except Exception as e:
        # Handle exceptions by printing the error and returning it
        print(f"Error making LLM: {e}")
        return f"Error making LLM: {e}"

def update_readme_and_create_pr(repo, updated_readme, readme_sha):
    # Define the commit message for the README update
    commit_message = "AI COMMIT: Proposed README update based on recent code changes."

    # Retrieve the commit SHA from environment variables and prepare the new branch name
    commit_sha = os.getenv('COMMIT_SHA')
    main_branch = repo.get_branch('main')
    new_branch_name = f'update-readme-{commit_sha[:7]}'
    # Create a new branch for the README update
    new_branch = repo.create_git_ref(ref=f'refs/heads/{new_branch_name}', sha=main_branch.commit.sha)

    # Update the README file in the new branch with the updated content
    repo.update_file("README.md", commit_message, updated_readme, readme_sha, branch=new_branch_name)

    # Create a pull request for the README update from the new branch to the main branch
    pr_title = "AI PR: Update README based on recent change done to the code"
    pr_body = "This is an AI PR. Please review the README"
    pull_request = repo.create_pull(title=pr_title, body=pr_body, head=new_branch_name, base="main")

    # Return the created pull request object
    return pull_request


