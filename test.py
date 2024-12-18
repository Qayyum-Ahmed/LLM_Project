import os
import json
import subprocess
import requests
import time

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise EnvironmentError("Please set the HUGGINGFACE_API_TOKEN environment variable.")

TEXT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"

MAX_RETRIES = 5          
RETRY_DELAY = 60        

def hf_inference_text(model: str, prompt: str) -> str:
    """Call the Hugging Face Inference API for a text model."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    else:
        return str(result)

def hf_inference_image(model: str, prompt: str) -> bytes:
    """Call the Hugging Face Inference API for an image model."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt, "image":"keyframes"}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.content  

def extract_json_after_answer(text):
    """Extract and parse the JSON object following the 'ANSWER:' keyword."""
    keyword = "ANSWER:"
    index = text.find(keyword)
    if index == -1:
        print("Keyword 'ANSWER:' not found in the text.")
        return None  

    substring_after = text[index + len(keyword):].strip()

    if substring_after.startswith("```"):
        end_fence = substring_after.find("```", 3)
        if end_fence != -1:
            substring_after = substring_after[3:end_fence].strip()

    try:
        return json.loads(substring_after)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None  

def ask_llm_for_plan(user_prompt):
    """Generate a video plan using the text model."""
    system_message = (
        f"You are a video production planner. This is the information provided by the user about the video:\n '{user_prompt}\n' "
        "You will output a plan in STRICT JSON format with no extra text.\n\n"
        "Make 10 frames per second. So technically JSON should contain num_key_frames=video_length*10 frames"
        "SCHEMA:\n"
        "{\n"
        "  \"duration_in_seconds\": <number_of_seconds>,\n"
        "  \"fps\": <frames_per_second>,\n"
        "  \"num_key_frames\": <number_of_key_frames>,\n"
        "  \"key_frames\": [\n"
        "    {\n"
        "      \"frame_index\": <int>,\n"
        "      \"description\": \"<DALL·E prompt>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "INSTRUCTIONS:\n"
        "- Try to add as much detail about each frame as possible so that it remains consistent throughout video. Also take into account smoothness since you have to generate 10 frames per second. So make sure that these frames make a smooth video\n"
        "- Respond with ONLY the JSON. No additional explanation.\n"
        "- Add some important information about the frame clearly in description as well. Like colour, direction, position of important things and setting in image."
        "- Choose a reasonable number of key frames for smooth interpolation.\n"
        "- Distribute frame_index values within total frames (duration_in_seconds * fps).\n"
        "- Description should be a detailed prompt for a text-to-image model, consistent style.\n"
        "- If you cannot produce a compliant response, return {}.\n"
        "ANSWER: "
    )

    print("Generating video plan using the text model...")
    content = hf_inference_text(TEXT_MODEL, system_message).strip()


    json_obj=content


    if json_obj is None:
        raise ValueError("Failed to extract a valid JSON plan from the model response.")

    print("Video plan successfully generated.")
    return json_obj

def generate_image_with_model(prompt, out_path):
    """Generate an image using the image model with retry on failure and log time taken."""
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            print(f"Generating image: '{out_path}' (Attempt {attempt + 1})")
            start_time = time.time()
            image_data = hf_inference_image(IMAGE_MODEL, prompt)
            elapsed_time = time.time() - start_time
            print(f"Image generation time: {elapsed_time:.2f} seconds")
            with open(out_path, 'wb') as f:
                f.write(image_data)
            print(f"Image saved to {out_path}")
            return
        except requests.HTTPError as e:
            status_code = e.response.status_code
            print(f"HTTP error occurred: {e}. Status Code: {status_code}")
            if status_code == 500:
                print(f"Service unavailable. Waiting for {RETRY_DELAY} seconds before retrying...")
                time.sleep(RETRY_DELAY)
                attempt += 1
            else:
                raise
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            attempt += 1

    raise RuntimeError(f"Failed to generate image after {MAX_RETRIES} attempts.")

def create_low_fps_video_from_keyframes(keyframes_dir, low_fps_video, duration, num_key_frames):
    """Create a low-FPS video from the keyframes using ffmpeg."""
    low_fps = num_key_frames / duration
    print(f"Creating low-FPS video at {low_fps:.2f} FPS...")
    frames = sorted([f for f in os.listdir(keyframes_dir) if f.endswith('.png')])
    temp_dir = os.path.join(keyframes_dir, "temp_ordered")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for i, frame in enumerate(frames):
        src = os.path.join(keyframes_dir, frame)
        dst = os.path.join(temp_dir, f"frame_{i+1:04d}.png")
        os.rename(src, dst)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-framerate", f"{low_fps:.2f}",
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        low_fps_video
    ], check=True)
    print(f"Low-FPS video created at {low_fps_video}")

    for filename in os.listdir(temp_dir):
        src = os.path.join(temp_dir, filename)
        dst = os.path.join(keyframes_dir, filename)
        os.rename(src, dst)
    os.rmdir(temp_dir)
    print("Temporary frame renaming completed.")

def interpolate_with_butterflow(low_fps_video, output_video, target_fps):
    """Interpolate frames using Butterflow to reach the desired FPS."""
    print(f"Interpolating video to {target_fps} FPS using Butterflow...")
    try:
        subprocess.run([
            "butterflow",
            "-o", output_video,
            "-r", str(target_fps),
            low_fps_video
        ], check=True)
        print(f"Interpolated video saved to {output_video}")
    except FileNotFoundError:
        raise FileNotFoundError("Butterflow executable not found. Please ensure Butterflow is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Butterflow failed with error: {e}")

def main(user_prompt):
    """Main workflow for generating the video."""
    try:
        plan = ask_llm_for_plan(user_prompt)
    except Exception as e:
        print(f"Error generating video plan: {e}")
        return

    duration = plan.get("duration_in_seconds")
    fps = plan.get("fps")
    num_key_frames = plan.get("num_key_frames")
    key_frames = plan.get("key_frames")

    if not all([duration, fps, num_key_frames, key_frames]):
        print("Invalid plan received. Please check the model's response.")
        return

    total_frames = int(duration * fps)
    print(f"Video Duration: {duration} seconds, FPS: {fps}, Total Frames: {total_frames}, Key Frames: {num_key_frames}")

    keyframes_dir = "keyframes"
    final_video_dir = "final_video"
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(final_video_dir, exist_ok=True)

    for kf in key_frames:
        idx = kf.get("frame_index")
        description = kf.get("description")
        if idx is None or description is None:
            print(f"Invalid key frame data: {kf}. Skipping...")
            continue

        out_path = os.path.join(keyframes_dir, f"frame_{idx:04d}.png")
        if os.path.exists(out_path):
            print(f"Frame {idx} already exists. Skipping image generation.")
            continue

        try:
            generate_image_with_model(description, out_path)
        except RuntimeError as e:
            print(f"Failed to generate image for frame {idx}: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while generating frame {idx}: {e}")
            return

    low_fps_video = os.path.join(keyframes_dir, "sparse_video.mp4")
    try:
        create_low_fps_video_from_keyframes(keyframes_dir, low_fps_video, duration, num_key_frames)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while creating low-FPS video: {e}")
        return

    output_video = os.path.join(final_video_dir, "output.mp4")
    try:
        interpolate_with_butterflow(low_fps_video, output_video, fps)
    except FileNotFoundError as e:
        print(e)
        return
    except RuntimeError as e:
        print(e)
        return
    except Exception as e:
        print(f"An unexpected error occurred during interpolation: {e}")
        return

    print("Video generation complete. Check final_video/output.mp4")

if __name__ == "__main__":
    user_prompt = (
        "Create a 5-second 30fps video of a white horse galloping through a lush green meadow at sunset. "
        "The horse should start on the left side of the frame and end on the right by the end of the video. "
        "Ensure the style is photorealistic, with warm lighting from the setting sun, and maintain a consistent camera angle (eye-level, wide-angle lens)."
    )
    main(user_prompt)
