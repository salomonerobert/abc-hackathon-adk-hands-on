import logging
from pydantic import BaseModel, Field
from google.adk.tools import ToolContext
from google import genai
from google.genai import types
from ..config import (
    VERTEX_AI_PROJECT,
    VERTEX_AI_LOCATION,
    PROMPT_REWRITE_MODEL,
    IMAGE_GENERATION_MODEL,
    VIDEO_GENERATION_MODEL,
    VIDEO_ASPECT_RATIO,
    VIDEO_NUMBER_OF_VIDEOS,
    VIDEO_DURATION_SECONDS,
    VIDEO_PERSON_GENERATION,
    VIDEO_RESOLUTION,
)

logger = logging.getLogger(__name__)

def get_next_version_number(tool_context: ToolContext, asset_name: str) -> int:
    """Get the next version number for a given asset name."""
    asset_versions = tool_context.state.get("asset_versions", {})
    current_version = asset_versions.get(asset_name, 0)
    next_version = current_version + 1
    return next_version

def update_asset_version(tool_context: ToolContext, asset_name: str, version: int, filename: str) -> None:
    """Update the version tracking for an asset."""
    if "asset_versions" not in tool_context.state:
        tool_context.state["asset_versions"] = {}
    if "asset_filenames" not in tool_context.state:
        tool_context.state["asset_filenames"] = {}

    tool_context.state["asset_versions"][asset_name] = version
    tool_context.state["asset_filenames"][asset_name] = filename

    # Also maintain a list of all versions for this asset
    asset_history_key = f"{asset_name}_history"
    if asset_history_key not in tool_context.state:
        tool_context.state[asset_history_key] = []
    tool_context.state[asset_history_key].append({"version": version, "filename": filename})

def create_versioned_filename(asset_name: str, version: int, file_extension: str = "png") -> str:
    """Create a versioned filename for an asset."""
    return f"{asset_name}_v{version}.{file_extension}"

class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="A detailed description of the image to generate.")
    aspect_ratio: str = Field(default="1:1", description="The desired aspect ratio, e.g., '1:1', '16:9'.")
    text_overlay: str = Field(default=None, description="Text to overlay on the image.")
    asset_name: str = Field(default="marketing_post", description="Base name for the marketing asset (will be versioned automatically).")

async def generate_image(tool_context: ToolContext, inputs: GenerateImageInput) -> str:
    """Generates a new image based on a prompt and other specifications."""
    # if "GEMINI_API_KEY" not in os.environ:
    #     raise ValueError("GEMINI_API_KEY environment variable not set.")

    print("Starting image generation: ")
    try:
        client = genai.Client(vertexai=True, project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)

        inputs = GenerateImageInput(**inputs)

        # Rewrite prompt for better image generation
        base_rewrite_prompt = f"""
        Rewrite the following prompt to be more descriptive and creative for an image generation model, adding relevant creative details: {inputs.prompt}
        **Important:** Output your prompt as a single paragraph"""

        if inputs.text_overlay:
            base_rewrite_prompt += f" the image should have the following text overlayed on it: '{inputs.text_overlay}'"

        prompt_rewrite_contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=base_rewrite_prompt)],
            ),
        ]

        rewritten_prompt_response = client.models.generate_content(
            model=PROMPT_REWRITE_MODEL,
            contents=prompt_rewrite_contents
        )
        rewritten_prompt = rewritten_prompt_response.text
        print(f"Rewritten prompt: {rewritten_prompt}")

        model = IMAGE_GENERATION_MODEL

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=rewritten_prompt)],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
            image_config=types.ImageConfig(
                aspect_ratio=inputs.aspect_ratio,
            ),
        )

        # Generate versioned filename for artifact
        version = get_next_version_number(tool_context, inputs.asset_name)
        artifact_filename = create_versioned_filename(inputs.asset_name, version)
        logger.info(f"Generating image with versioned artifact filename: {artifact_filename} (version {version})")

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data

                # Create a Part object from the inline data to save as artifact
                image_part = types.Part(inline_data=inline_data)

                try:
                    # Save the image as an artifact
                    version = await tool_context.save_artifact(
                        filename=artifact_filename,
                        artifact=image_part
                    )

                    # Update version tracking
                    update_asset_version(tool_context, inputs.asset_name, version, artifact_filename)

                    # Store artifact filename in session state for future reference
                    tool_context.state["last_generated_image"] = artifact_filename
                    tool_context.state["current_asset_name"] = inputs.asset_name

                    logger.info(f"Saved generated image as artifact '{artifact_filename}' (version {version})")

                    return f"Image generated successfully! Saved as artifact: {artifact_filename} (version {version} of {inputs.asset_name})"

                except Exception as e:
                    logger.error(f"Error saving artifact: {e}")
                    return f"Error saving generated image as artifact: {e}"
            else:
                print(chunk.text)

        return "No image was generated"
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while generating the image: {e}"