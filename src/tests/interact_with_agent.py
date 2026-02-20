"""
Interactive test script for Trail Guide Agent.
Allows you to chat with the agent from the terminal.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.projects import AIProjectClient


def load_env_with_fallbacks(env_path: Path) -> None:
    encodings = [None, "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]
    last_error = None

    for encoding in encodings:
        try:
            if encoding is None:
                load_dotenv(env_path)
            else:
                load_dotenv(env_path, encoding=encoding)
            return
        except UnicodeDecodeError as ex:
            last_error = ex

    raise RuntimeError(f"Unable to read .env file at {env_path}. Please save it as UTF-8.") from last_error


# Load environment variables from repository root
repo_root = Path(__file__).parent.parent.parent
env_file = repo_root / '.env'
load_env_with_fallbacks(env_file)

required_vars = ["AZURE_AI_PROJECT_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "AGENT_NAME"]
missing_vars = [name for name in required_vars if not os.getenv(name)]
if missing_vars:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(missing_vars)}. "
        f"Expected .env at: {env_file}"
    )

def interact_with_agent():
    """Start an interactive chat session with the Trail Guide Agent."""
    
    # Initialize project client
    project_client = AIProjectClient(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    
    # Get agent name from environment or use default
    agent_name = os.getenv("AGENT_NAME", "trail-guide")

    # Resolve latest agent definition (model + instructions)
    agent = project_client.agents.get(agent_name=agent_name)
    latest_version = agent.versions.latest
    definition = latest_version.definition
    model_name = (
        os.getenv("MODEL_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("MODEL_NAME")
        or definition.model
    )
    instructions = definition.instructions

    # Runtime client for chat completion (direct Azure OpenAI endpoint)
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    chat_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21",
    )
    messages = [{"role": "system", "content": instructions}]
    
    print(f"\n{'='*60}")
    print(f"Trail Guide Agent - Interactive Chat")
    print(f"Agent: {agent_name} (version: {latest_version.version})")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print("\nType your questions or requests. Type 'exit' or 'quit' to end the session.\n")
    
    print("Started conversation.\n")
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nEnding session. Goodbye!")
                break
            
            # Send message with conversation history
            messages.append({"role": "user", "content": user_input})
            response = chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

            assistant_reply = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_reply})
            print(f"\nAgent: {assistant_reply}\n")
                    
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Goodbye!")
    except Exception as e:
        error_text = str(e)
        if "DeploymentNotFound" in error_text or "Error code: 404" in error_text:
            print("\nError: Model deployment not found.")
            print(f"Requested model/deployment: {model_name}")
            print("Target endpoint: AZURE_OPENAI_ENDPOINT")

            print("\nFix:")
            print("1) Deploy a chat model in your AI Services account.")
            print("2) Set MODEL_DEPLOYMENT in .env to that deployment name.")
            print("3) Re-run this test script.")
        else:
            print(f"\nError: {e}")
        sys.exit(1)
    finally:
        project_client.close()

if __name__ == "__main__":
    interact_with_agent()
