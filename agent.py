from google.adk.agents import LlmAgent

from config import AGENT_MODEL
from prompt import SOCIAL_MEDIA_AGENT_PROMPT

root_agent = LlmAgent(
    name="root_agent",
    model=AGENT_MODEL,
    instruction=SOCIAL_MEDIA_AGENT_PROMPT,
    tools=[generate_image, edit_image, generate_video, load_artifacts_tool],
)