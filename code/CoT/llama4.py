#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import numpy as np
from openai import OpenAI
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


def get_frames_from_folder(folder_path, num_frames=6):

    frame_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        frame_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not frame_files:
        print(f"警告: 在 {folder_path} 中没有找到图片")
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

    base_prompt = ("This is a video that may contain harmful content such as hate speech, explicit violence, discrimination, or other offensive behavior. "
                   "You are a content moderation expert. Analyze this video using a reasoning process. The video is represented by:  Visual frames and Transcript text (could be none). "
                   "HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor. OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech. IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech."
                   "Please reason in four steps, then summarize your final judgment. "
                   "Step 1: Visual Understanding. Describe the visual content in the frames. Focus on characters, scenes, and potential symbolic or hateful imagery.  "
                   "Step 2: Textual Understanding.  Analyze the transcript of the video.  Pay attention to metaphors, puns, homophones, or rhetorical devices that may express hate or implicit bias. "
                   "Step 3: Cross-Modal Reasoning. Integrate insights from the visuals and the transcript.  Identify any **implicit meanings**, contradictions, or cross-modal cues that enhance or modify the hateful nature of the content. "
                   "Step 4: Final Evaluation. Does this video express harmful or hateful content? Return with label: 0 or 1, while 1 is hate or offensive, 0 is non-hate. Just return 1/0, no extra word. "
                   "Keep the answer concise, structured(following these four steps), and professional."
                   )

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


def classify_multimodal_content_single_turn(video_key, text, client, video_folder, turn, previous_results=None,
                                            num_frames=16):

    frames_paths = get_frames_from_folder(video_folder, num_frames)

    if not frames_paths:
        print(f"Warning: No image frames found for video {video_key}")
        return None

    text = clean_text(text)


    user_prompt = create_prompt(text)


    content = [{"type": "text", "text": user_prompt}]


    for frame_path in frames_paths:
        base64_image = encode_image_to_base64(frame_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="LLM-Research/Llama-4-Maverick-17B-128E-Instruct",
                messages=[{
                    'role': 'user',
                    'content': content
                }],
                stream=False
            )

            break

        except Exception as e:
            error_message = str(e)


            if "data_inspection_failed" in error_message:

                return {"error": "data_inspection_failed", "assumed_hate": True}


            elif "insufficient_quota" in error_message or "429" in error_message:
                wait_time = 2 ** attempt  # 2, 4, 8, 16, 32

                time.sleep(wait_time)
            else:
                print(f" {error_message}")

                return None
    else:
        print(f"After multiple failed retries, the sample was skipped: {video_key}")
        return None


    result_text = response.choices[0].message.content.strip()
    return result_text


def process_video_single(video_key, video_folder, transcript, client, num_frames=6):

    print(f"\n{'=' * 80}")
    print(f"Processing video: {video_key}")
    print(f"Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    print(f"{'=' * 80}")


    raw_response = classify_multimodal_content_single_turn(
        video_key, transcript, client, video_folder, 1, None, num_frames
    )

    if raw_response is None:

        return {
            'video_id': video_key,
            'status': 'error',
            'error': 'API call failed',
            'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
            'transcript': transcript,
            'response': None
        }


    if isinstance(raw_response, dict) and raw_response.get("error") == "data_inspection_failed":
        return {
            'video_id': video_key,
            'status': 'success',
            'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
            'transcript': transcript,
            'response': "Data inspection failed - assumed hate content"
        }


    print(f"\n📝 Model Response:")
    print("🔹" + "🔹" * 25)
    print(raw_response)
    print("🔹" + "🔹" * 25)


    final_result = {
        'video_id': video_key,
        'status': 'success',
        'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
        'transcript': transcript,
        'response': raw_response
    }

    return final_result


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


def process_all_videos(video_frames_dir: str, transcript_json: str, output_file: str, filter_type: str = "all"):


    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='apikey',
    )


    transcripts = load_audio_transcripts(transcript_json)


    existing_results = []
    completed_video_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)


            for result in existing_results:
                video_id = result.get('video_id')
                if video_id and result.get('status') == 'success':
                    completed_video_ids.add(video_id)

            print(f"Found {len(completed_video_ids)} videos already completed")

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
    elif filter_type == "all":
        print(f"Processing all {len(video_folders)} videos")


    videos_to_process = [vf for vf in video_folders if vf.name not in completed_video_ids]
    print(f"Need to process {len(videos_to_process)} videos (skipping {len(completed_video_ids)} completed)")

    results = existing_results.copy()

    for video_folder in tqdm(videos_to_process, desc=f"Processing {filter_type} videos"):
        video_id = video_folder.name
        transcript = transcripts.get(video_id, "")

        try:
            result = process_video_single(video_id, str(video_folder), transcript, client)
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
                'response': None
            })


    _save_results(results, output_file)
    print(f"Processing completed. Results saved to {output_file}")
    print(f"Total videos processed: {len(results)}")
    print(f"Newly processed: {len(videos_to_process)}")
    print(f"Previously completed: {len(completed_video_ids)}")


def main():

    video_frames_dir = ""


    transcript_json = ""


    output_file = "hate_detection_hatemm_results_CoT_with_label_llama4.json"


    filter_type = "all"



    if not os.path.exists(video_frames_dir):
        print(f"Video frames directory not found: {video_frames_dir}")
        return

    if not os.path.exists(transcript_json):
        print(f"Transcript JSON file not found: {transcript_json}")
        return

    print(f"Processing videos with filter_type: {filter_type}")
    print(f"Video frames directory: {video_frames_dir}")
    print(f"Transcript file: {transcript_json}")
    print(f"Output file: {output_file}")


    process_all_videos(
        video_frames_dir,
        transcript_json,
        output_file,
        filter_type
    )


if __name__ == "__main__":
    main()