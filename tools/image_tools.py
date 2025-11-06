import asyncio
import logging
from pydantic import BaseModel, Field
from google.adk.tools import ToolContext
from google import genai
from google.genai import types
import os
from abc_hackathon_agent_hands_on.config import (
    VERTEX_AI_PROJECT,
    VERTEX_AI_LOCATION,
    IMAGE_GENERATION_MODEL,
    PROMPT_REWRITE_MODEL,
    VIDEO_GENERATION_MODEL,
    VIDEO_ASPECT_RATIO,
    VIDEO_DURATION_SECONDS,
    VIDEO_NUMBER_OF_VIDEOS,
    VIDEO_PERSON_GENERATION,
    VIDEO_RESOLUTION
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

class EditImageInput(BaseModel):
    artifact_filename: str = Field(..., description="The filename of the image artifact to edit.")
    prompt: str = Field(..., description="The prompt describing only the desired changes to be made to the image.")
    aspect_ratio: str = Field(..., description="Aspect ratio of output image in string format. Accepted values are '1:1', '16:9', '9:16', '4:5', '5:4', '2:3', '3:2'.")
    asset_name: str = Field(default=None, description="Optional: specify asset name for the new version (defaults to incrementing current asset).")

class GenerateVideoInput(BaseModel):
    prompt: str = Field(..., description="A detailed description of the video to generate, i.e. how to animate the reference image.")
    asset_name: str = Field(default="marketing_post", description="Base name for the marketing asset (will be versioned automatically).")
    reference_image_filename: str = Field(..., description="Filename of a reference image to animate. Use 'latest' to use the most recently generated image.")

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
    
async def edit_image(tool_context: ToolContext, inputs: EditImageInput) -> str:
    """Edits an existing image based on a prompt."""
    # if "GEMINI_API_KEY" not in os.environ:
    #     raise ValueError("GEMINI_API_KEY environment variable not set.")

    print("Starting image edit")

    try:
        inputs = EditImageInput(**inputs)

        # Load the image artifact
        logger.info(f"Loading artifact: {inputs.artifact_filename}")
        try:
            loaded_image_part = await tool_context.load_artifact(inputs.artifact_filename)
            logger.info("Loaded image artifact successfully")
            if not loaded_image_part:
                return f"Could not find image artifact: {inputs.artifact_filename}"
        except Exception as e:
            logger.error(f"Error loading artifact: {e}")
            return f"Error loading image artifact: {e}"

        client = genai.Client(vertexai=True, project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)

        model = IMAGE_GENERATION_MODEL

        contents = [
            types.Content(
                role="user",
                parts=[loaded_image_part, types.Part.from_text(text=inputs.prompt)],
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

        # Determine asset name and generate versioned filename
        if inputs.asset_name:
            asset_name = inputs.asset_name
        else:
            # Try to extract asset name from current artifact filename
            current_asset_name = tool_context.state.get("current_asset_name")
            if current_asset_name:
                asset_name = current_asset_name
            else:
                # Fallback: extract from filename if it follows our versioning pattern
                base_name = inputs.artifact_filename.split('_v')[0] if '_v' in inputs.artifact_filename else "marketing_post"
                asset_name = base_name

        version = get_next_version_number(tool_context, asset_name)
        edited_artifact_filename = create_versioned_filename(asset_name, version)
        logger.info(f"Editing image with versioned artifact filename: {edited_artifact_filename} (version {version})")
        
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
                edited_image_part = types.Part(inline_data=inline_data)

                try:
                    # Save the edited image as an artifact
                    version = await tool_context.save_artifact(
                        filename=edited_artifact_filename,
                        artifact=edited_image_part
                    )

                    # Update version tracking
                    update_asset_version(tool_context, asset_name, version, edited_artifact_filename)

                    # Store artifact filename in session state for future reference
                    tool_context.state["last_generated_image"] = edited_artifact_filename
                    tool_context.state["current_asset_name"] = asset_name

                    logger.info(f"Saved edited image as artifact '{edited_artifact_filename}' (version {version})")

                    return f"Image edited successfully! Saved as artifact: {edited_artifact_filename} (version {version} of {asset_name})"

                except Exception as e:
                    logger.error(f"Error saving edited artifact: {e}")
                    return f"Error saving edited image as artifact: {e}"
            else:
                print(chunk.text)

        return "No edited image was generated"
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while editing the image: {e}"
    
async def generate_video(tool_context: ToolContext, inputs: GenerateVideoInput) -> str:
    """Generates a new video by animating a reference image based on a prompt."""
    # if "GEMINI_API_KEY" not in os.environ:
    #     raise ValueError("GEMINI_API_KEY environment variable not set.")

    print("Starting video generation: ")
    try:
        client = genai.Client(
            vertexai=True, project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION,
        )

        inputs = GenerateVideoInput(**inputs)

        # Handle reference image
        if inputs.reference_image_filename == "latest":
            ref_filename = tool_context.state.get("last_generated_image")
        else:
            ref_filename = inputs.reference_image_filename

        if not ref_filename:
            return "No reference image specified or found. Please specify a reference_image_filename or generate an image first."

        # Load the reference image
        try:
            reference_image_part = await tool_context.load_artifact(ref_filename)
            if not reference_image_part:
                return f"Could not load reference image: {ref_filename}"
        except Exception as e:
            logger.error(f"Error loading reference image {ref_filename}: {e}")
            return f"Error loading reference image: {e}"

        logger.info(f"Using reference image: {ref_filename}")
        image_bytes = reference_image_part.inline_data.data
        mime_type = reference_image_part.inline_data.mime_type

        # Rewrite prompt
        base_rewrite_prompt = f"""
        Rewrite the following prompt to be more descriptive and creative for a video generation model that animates a still image.
        The goal is to bring the image to life. Add details about movement, atmosphere, and focus.
        Original prompt: {inputs.prompt}
        **Important:** Output your prompt as a single paragraph.
        """
        prompt_rewrite_content_parts = [
            types.Part.from_text(text=base_rewrite_prompt),
            reference_image_part
        ]
        prompt_rewrite_contents = [
            types.Content(
                role="user",
                parts=prompt_rewrite_content_parts,
            ),
        ]
        rewritten_prompt_response = client.models.generate_content(model=PROMPT_REWRITE_MODEL, contents=prompt_rewrite_contents)
        rewritten_prompt = rewritten_prompt_response.text
        print(f"Rewritten prompt: {rewritten_prompt}")

        # Video generation
        video_asset_name = f"{inputs.asset_name}_video"
        version = get_next_version_number(tool_context, video_asset_name)
        artifact_filename = create_versioned_filename(video_asset_name, version, file_extension="mp4")
        logger.info(f"Generating video with versioned artifact filename: {artifact_filename} (version {version})")

        video_config = types.GenerateVideosConfig(
            aspect_ratio=VIDEO_ASPECT_RATIO,
            number_of_videos=VIDEO_NUMBER_OF_VIDEOS,
            duration_seconds=VIDEO_DURATION_SECONDS,
            person_generation=VIDEO_PERSON_GENERATION,
            resolution=VIDEO_RESOLUTION,
        )

        operation = client.models.generate_videos(
            model=VIDEO_GENERATION_MODEL,
            prompt=rewritten_prompt,
            image=types.Image(image_bytes=image_bytes, mime_type=mime_type),
            config=video_config,
        )

        print("Video generation started. Waiting for completion...")
        while not operation.done:
            print("Video has not been generated yet. Check again in 10 seconds...")
            await asyncio.sleep(10)
            operation = client.operations.get(operation)

        result = operation.result
        if not result or not result.generated_videos:
            logger.error("Video generation failed or produced no videos.")
            return "Error occurred while generating video, or no videos were generated."

        print(f"Generated {len(result.generated_videos)} video(s).")

        # Process the first generated video
        generated_video = result.generated_videos[0]

        temp_video_filename = f"temp_{artifact_filename}"
        try:
            print(f"Downloading generated video: {generated_video.video.uri}")
            generated_video.video.save(temp_video_filename)
            print(f"Video downloaded to temporary file: {temp_video_filename}")

            with open(temp_video_filename, "rb") as f:
                video_bytes = f.read()

            video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

            saved_version = await tool_context.save_artifact(
                filename=artifact_filename,
                artifact=video_part
            )

            update_asset_version(tool_context, video_asset_name, saved_version, artifact_filename)

            tool_context.state["last_generated_video"] = artifact_filename
            tool_context.state["current_asset_name"] = video_asset_name

            logger.info(f"Saved generated video as artifact '{artifact_filename}' (version {saved_version})")
            return f"Video generated successfully! Saved as artifact: {artifact_filename} (version {saved_version} of {video_asset_name})"

        except Exception as e:
            logger.error(f"Error saving video artifact: {e}")
            return f"Error saving generated video as artifact: {e}"
        finally:
            if os.path.exists(temp_video_filename):
                os.remove(temp_video_filename)

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while generating the video: {e}"