"""Configuration file for ABC Hackathon Agent models and settings."""

# Vertex AI Configuration
VERTEX_AI_PROJECT = "ebc-demos"
VERTEX_AI_LOCATION = "us-central1"

# Model Names
AGENT_MODEL = "gemini-2.5-flash"
PROMPT_REWRITE_MODEL = "gemini-2.5-flash"
IMAGE_GENERATION_MODEL = "gemini-2.5-flash-image"
VIDEO_GENERATION_MODEL = "veo-2.0-generate-001"

# Video Generation Defaults
VIDEO_ASPECT_RATIO = "16:9"
VIDEO_NUMBER_OF_VIDEOS = 1
VIDEO_DURATION_SECONDS = 8
VIDEO_PERSON_GENERATION = "ALLOW_ADULT"
VIDEO_RESOLUTION = "720p"
