import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition


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
repo_root = Path(__file__).resolve().parents[3]
env_file = repo_root / '.env'
load_env_with_fallbacks(env_file)

required_vars = ["AZURE_AI_PROJECT_ENDPOINT", "AGENT_NAME"]
missing_vars = [name for name in required_vars if not os.getenv(name)]
if missing_vars:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(missing_vars)}. "
        f"Expected .env at: {env_file}"
    )

# Read instructions from prompt file
prompt_file = Path(__file__).parent / 'prompts' / 'v2_instructions.txt'
with open(prompt_file, 'r') as f:
    instructions = f.read().strip()

project_client = AIProjectClient(
    endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = project_client.agents.create_version(
    agent_name=os.environ["AGENT_NAME"],
    definition=PromptAgentDefinition(
        model=os.getenv("MODEL_NAME", "gpt-4.1"),  # Use Global Standard model
        instructions=instructions,
    ),
)
print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")