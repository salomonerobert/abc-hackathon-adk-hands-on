from google.adk.agents import LlmAgent
from google.adk.tools.load_artifacts_tool import load_artifacts_tool

from abc_hackathon_agent_hands_on.config import AGENT_MODEL
from abc_hackathon_agent_hands_on.prompt import SOCIAL_MEDIA_AGENT_PROMPT
from abc_hackathon_agent_hands_on.tools.image_tools import edit_image, generate_image, generate_video

root_agent = LlmAgent(
    name="root_agent",
    model=AGENT_MODEL,
    instruction=SOCIAL_MEDIA_AGENT_PROMPT,
    tools=[generate_image, edit_image, generate_video, load_artifacts_tool],
)