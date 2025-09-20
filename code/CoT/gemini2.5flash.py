import os
import base64
import numpy as np
import google.generativeai as genai
import json
from tqdm import tqdm
import glob
import time
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional


def clean_text(text):

    return text.strip() if text else ""


def encode_image_to_base64(image_path):

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:

        return None


def load_image_for_gemini(image_path):

    try:
        from PIL import Image
        image = Image.open(image_path)
        return image
    except Exception as e:

        return None


def get_frames_from_folder(folder_path, num_frames=6):

    frame_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        frame_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not frame_files:
        print(f"Warning: Image not found in {folder_path}")
        return []


    frame_files.sort()


    if len(frame_files) > num_frames:

        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    elif len(frame_files) < num_frames:

        while len(frame_files) < num_frames:
            frame_files.append(frame_files[len(frame_files) % len(frame_files)])

    return frame_files


def create_prompt(transcript: str = "") -> str:

    base_prompt = (
        "This is a video that may contain harmful content such as hate speech, explicit violence, discrimination, or other offensive behavior. "
        "You are a content moderation expert. Analyze this video using a reasoning process. The video is represented by: Visual frames and Transcript text (could be none). "
        "HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor. OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech. IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech."
        "Please reason in four steps, then summarize your final judgment. "
        "Step 1: Visual Understanding. Describe the visual content in the frames. Focus on characters, scenes, and potential symbolic or hateful imagery. "
        "Step 2: Textual Understanding. Analyze the transcript of the video. Pay attention to metaphors, puns, homophones, or rhetorical devices that may express hate or implicit bias. "
        "Step 3: Cross-Modal Reasoning. Integrate insights from the visuals and the transcript. Identify any **implicit meanings**, contradictions, or cross-modal cues that enhance or modify the hateful nature of the content. "
        "Step 4: Final Evaluation. Does this video express harmful or hateful content? Return with label: 0 or 1, while 1 is hate or offensive, 0 is non-hate. Just return 1/0, no extra word. "
        "Keep the answer concise, structured(following these four steps), and professional.")

    if transcript.strip():
        prompt = base_prompt + f"\n\nTranscript: {transcript}"
    else:
        prompt = base_prompt + "\n\nTranscript: No transcript available."

    return prompt


def clean_json_response(response_text: str) -> tuple:

    try:
        cleaned = response_text.strip()


        if cleaned.startswith('```json') and cleaned.endswith('```'):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = cleaned[3:-3].strip()


        parsed_json = json.loads(cleaned)
        return parsed_json, True

    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return response_text, False


def classify_multimodal_content_cot(video_key, text, model, video_folder, num_frames=16):

    frames_paths = get_frames_from_folder(video_folder, num_frames)

    if not frames_paths:
        print(f"Warning: No image frames found for video {video_key}")
        return None

    text = clean_text(text)


    user_prompt = create_prompt(text)


    contents = [user_prompt]


    for frame_path in frames_paths:
        image = load_image_for_gemini(frame_path)
        if image:
            contents.append(image)

    max_retries = 5
    last_response = None

    for attempt in range(max_retries):
        try:

            response = model.generate_content(contents)

            last_response = response


            if response and response.text:
                result_text = response.text
                return result_text
            else:
                print(f"[Error] API call returns an empty response")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[Retry] {attempt + 1}th failure, waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    break

        except Exception as e:
            error_message = str(e)
            print(f"[Exception] Exception raised: {error_message}")


            if "safety" in error_message.lower() or "block" in error_message.lower() or "candidate" in error_message.lower():
                print(f"[Content Filtering] Sample {video_key}, flagged as hateful content")
                return {"error": "content_policy_violation", "assumed_hate": True}


            elif "rate_limit" in error_message.lower() or "quota" in error_message.lower():
                wait_time = 2 ** attempt
                print(f"[Rate limiting] {attempt + 1}th retry, waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[Other Anomalies] Sample: {video_key}")
                if attempt == max_retries - 1:

                    return {"error": "api_exception", "error_message": error_message}


    print(f"[Failure] All retries have failed. Sample: {video_key}")
    return {"error": "all_retries_failed", "error_message": "All failed"}


def process_video_cot(video_key, video_folder, transcript, model, num_frames=16):

    print(f"\n{'=' * 80}")
    print(f"Processing video: {video_key}")
    print(f"Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    print(f"{'=' * 80}")


    raw_response = classify_multimodal_content_cot(
        video_key, transcript, model, video_folder, num_frames
    )


    if raw_response is None:

        result = {
            'video_id': video_key,
            'status': 'error',
            'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
            'transcript': transcript,
            'cot_response': {
                'raw_response': None,
                'parsed_response': None,
                'parse_success': False,
                'error': 'No response received'
            }
        }

        return result


    if isinstance(raw_response, dict) and "error" in raw_response:
        error_type = raw_response.get("error")

        if error_type == "content_policy_violation":
            result = {
                'video_id': video_key,
                'status': 'content_policy_violation',
                'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
                'transcript': transcript,
                'cot_response': {
                    'raw_response': "Content policy violation",
                    'parsed_response': None,
                    'parse_success': False,
                    'error': 'content_policy_violation'
                }
            }

            return result

        else:
            # 其他类型的错误
            raw_content = raw_response.get("raw_response", str(raw_response))

            print(f"\n📝 Gemini Response - Error response:")
            print("🔹" + "🔹" * 25)
            print(f"Error type: {error_type}")
            print(f"Raw response: {raw_content}")
            print("🔹" + "🔹" * 25)

            result = {
                'video_id': video_key,
                'status': 'error',
                'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
                'transcript': transcript,
                'cot_response': {
                    'raw_response': raw_content,
                    'parsed_response': None,
                    'parse_success': False,
                    'error': error_type,
                    'error_details': raw_response
                }
            }
            return result


    if isinstance(raw_response, str):

        print(f"\n📝 Gemini CoT Response:")
        print("🔹" + "🔹" * 25)
        print(raw_response)
        print("🔹" + "🔹" * 25)


        result = {
            'video_id': video_key,
            'status': 'success',
            'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
            'transcript': transcript,
            'cot_response': {
                'raw_response': raw_response,
                'parsed_response': raw_response,
                'parse_success': True
            }
        }

    else:
        # 其他未预期的响应类型
        print(f"\n📝 Gemini Response - Unexpected response type:")
        print("🔹" + "🔹" * 25)
        print(f"Response type: {type(raw_response)}")
        print(f"Response content: {raw_response}")
        print("🔹" + "🔹" * 25)

        result = {
            'video_id': video_key,
            'status': 'error',
            'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
            'transcript': transcript,
            'cot_response': {
                'raw_response': str(raw_response),
                'parsed_response': None,
                'parse_success': False,
                'error': 'unexpected_response_type',
                'response_type': str(type(raw_response))
            }
        }

    return result


def load_audio_transcripts(json_path: str) -> Dict[str, str]:

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded transcripts for {len(data)} videos")
        return data
    except Exception as e:
        print(f"Error loading audio transcripts: {e}")
        return {}


def _save_results(results: List[Dict], output_file: str):

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def process_all_videos(video_frames_dir: str, transcript_json: str, output_file: str, filter_type: str = "all",
                       filter_count: int = 500, filter_position: str = "first"):


    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:

        api_key = ""
        print("Warning: Please set the GEMINI_API_KEY environment variable or configure the API Key directly within your code.")


    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')


    transcripts = load_audio_transcripts(transcript_json)


    existing_results = []
    completed_video_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            for result in existing_results:
                video_id = result.get('video_id')
                cot_response = result.get('cot_response', {})

                if (video_id and
                        cot_response and
                        'raw_response' in cot_response and
                        result.get('status') == 'success'):
                    completed_video_ids.add(video_id)

            print(f"Found {len(completed_video_ids)} videos already completed with Gemini CoT")

        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = []


    video_folders = [d for d in Path(video_frames_dir).iterdir() if d.is_dir()]
    video_folders.sort()

    print(f"Found {len(video_folders)} video folders")

    if filter_type == "hate":
        video_folders = [vf for vf in video_folders if vf.name.startswith("hate")]
        print(f"Filtered to {len(video_folders)} hate videos")
    elif filter_type == "non_hate":
        video_folders = [vf for vf in video_folders if vf.name.startswith("non_hate")]
        print(f"Filtered to {len(video_folders)} non-hate videos")
    elif filter_type == "first_n":

        total_videos = len(video_folders)
        video_folders = video_folders[:filter_count]
        print(f"Selected first {len(video_folders)} videos (out of {total_videos})")
    elif filter_type == "exclude_first_n":

        total_videos = len(video_folders)
        video_folders = video_folders[filter_count:]
        print(f"Selected {len(video_folders)} videos (excluding first {filter_count} out of {total_videos})")
    elif filter_type == "all":
        print(f"Processing all {len(video_folders)} videos")
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}")


    videos_to_process = [vf for vf in video_folders if vf.name not in completed_video_ids]
    print(f"Need to process {len(videos_to_process)} videos (skipping {len(completed_video_ids)} completed)")


    if videos_to_process:
        print(f"First video to process: {videos_to_process[0].name}")
        print(f"Last video to process: {videos_to_process[-1].name}")

    results = existing_results.copy()

    for video_folder in tqdm(videos_to_process, desc=f"Processing videos with Gemini CoT analysis"):
        video_id = video_folder.name
        transcript = transcripts.get(video_id, "")

        try:
            result = process_video_cot(video_id, str(video_folder), transcript, model)
            results.append(result)


            if len(results) % 10 == 0:
                _save_results(results, output_file)

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'video_id': video_id,
                'status': 'error',
                'error': str(e),
                'cot_response': None
            })


    _save_results(results, output_file)
    print(f"Gemini CoT processing completed. Results saved to {output_file}")
    print(f"Total videos in results: {len(results)}")
    print(f"Newly processed: {len(videos_to_process)}")
    print(f"Previously completed: {len(completed_video_ids)}")


def main():

    video_frames_dir = "./Video_frames_CLIP"

    transcript_json = ""


    output_file = "cot_hate_hatemm_detection_results_gemini25flash.json"

    filter_type = "all"
    filter_count = 500  # This parameter is not used in all mode.


    if not os.path.exists(video_frames_dir):
        print(f"Video frames directory not found: {video_frames_dir}")
        return

    if not os.path.exists(transcript_json):
        print(f"Transcript JSON file not found: {transcript_json}")
        return

    print(f"Processing videos with Gemini 2.5 Flash CoT configuration:")
    print(f"  Filter type: {filter_type}")
    if filter_type in ["first_n", "exclude_first_n"]:
        print(f"  Count: {filter_count}")
    print(f"  Video frames directory: {video_frames_dir}")
    print(f"  Transcript file: {transcript_json}")
    print(f"  Output file: {output_file}")


    process_all_videos(
        video_frames_dir,
        transcript_json,
        output_file,
        filter_type,
        filter_count
    )


if __name__ == "__main__":
    main()