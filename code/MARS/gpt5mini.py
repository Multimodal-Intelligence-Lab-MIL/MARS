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
        print(f"Unable to encode image {image_path}: {e}")
        return None


def get_frames_from_folder(folder_path, num_frames=16):

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


def create_turn_prompt(turn: int, transcript: str = "", previous_results: Dict = None) -> str:


    if turn == 1:

        base_prompt = """You are a content analysis expert. Provide a detailed, objective description of this video content.

Task: Describe what you observe in the video frames and transcript:
1. objective_visual_description: Describe what you see - objects, people, actions, settings, text, symbols, colors, expressions, gestures, etc.

Rules:
- Be comprehensive and detailed
- NO analysis, interpretations, or value judgments
- Simply report what you observe
- Return ONLY valid JSON with one key: ["objective_visual_description"]"""

    elif turn == 2:

        base_prompt = """You are a hateful content detection specialist. Analyze the video and transcript content under the ASSUMPTION that it contains hateful or offensive content.

HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech.

Task: Identify evidence supporting hateful(offensive) interpretation:
1. evidence: Describe specific evidence supporting the hateful nature of the content
2. reasoning: Explain why it is hateful content.
3. strength: Your confidence in this hate interpretation [0-1]

Focus on:
- Language targeting religion, ethnicity, race, gender, nationality, etc.
- Group-based dehumanizing language or symbols
- Identity-based discriminatory attacks (not personal insults)
- Incitement against identity groups

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""

    elif turn == 3:

        base_prompt = """You are a content analyst. Analyze this content under the ASSUMPTION that it does NOT constitute hate speech and offensive content.

HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech.

Task: Identify evidence supporting non-hate(non-offensive) interpretation:
1. evidence: Describe specific evidence supporting the non-hateful nature of the content
2. reasoning: Explain why this content does not hateful
3. strength: Your confidence in this non-hate interpretation [0-1]

Consider:
- Is this a personal dispute rather than group targeting?
- Are insults directed at individuals rather than identity groups?
- Is there artistic, satirical, or educational context?
- Does the content lack group-based discriminatory language?

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""

    elif turn == 4:

        base_prompt = f"""You are a senior content moderation specialist making the final determination.

OBJECTIVE DESCRIPTION: {json.dumps(previous_results.get('turn_1', {}).get('parsed_response', {}), ensure_ascii=False)}

HATE ARGUMENT: {json.dumps(previous_results.get('turn_2', {}).get('parsed_response', {}), ensure_ascii=False)}

NON-HATE ARGUMENT: {json.dumps(previous_results.get('turn_3', {}).get('parsed_response', {}), ensure_ascii=False)}

Task: Conduct critical analysis and make final determination:

1. evidence_comparison: Compare the quality and strength of both arguments, identifying which evidence is more compelling
2. contextual_analysis: Analyze how the content's context (platform, audience, intent) affects interpretation
3. harm_assessment: Evaluate potential real-world impact and harm to targeted groups
4. final_decision:
   - label: 0 (non-hate) or 1 (hate/offensive)
   - confidence: Overall confidence in decision [0-1]
   - key_factors: The decisive elements that determined your judgment
   - reasoning: 2-3 sentences explaining your decision

Rules:
- Weigh evidence objectively, not just confidence scores
- Consider both explicit and subtle indicators
- Prioritize potential for real-world harm
- Base decision on strongest evidence, not balanced arguments

Return ONLY valid JSON with key "final_decision" containing the above structure"""

    else:
        raise ValueError(f"Invalid turn number: {turn}")


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
    """
    Performing single-round inference on video image frames and text using OpenAI GPT-5-mini
    """
    frames_paths = get_frames_from_folder(video_folder, num_frames)

    if not frames_paths:
        print(f"Warning: No image frames found for video {video_key}")
        return None

    text = clean_text(text)


    user_prompt = create_turn_prompt(turn, text, previous_results)


    content = [{"type": "text", "text": user_prompt}]

    # Add image - Modified to OpenAI format
    for frame_path in frames_paths:
        base64_image = encode_image_to_base64(frame_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            })

    messages = [{
        "role": "user",
        "content": content
    }]

    max_retries = 5
    last_response = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                max_completion_tokens=4096,
                #temperature=0.1
            )

            last_response = response

            if response and response.choices:
                result_text = response.choices[0].message.content
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
            print(f"[Exception] Call exception: {error_message}")


            if "content_policy_violation" in error_message or "safety" in error_message.lower():
                print(f"[Content Filtering] Sample {video_key} Round {turn}, flagged as hateful content")
                return {"error": "content_policy_violation", "assumed_hate": True}


            elif "rate_limit" in error_message.lower() or "429" in error_message or "quota" in error_message.lower():
                wait_time = 2 ** attempt
                print(f"[Rate limiting] {attempt + 1}th retry, waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[Other Anomalies] Sample: {video_key}, Turn: {turn}")
                if attempt == max_retries - 1:

                    return {"error": "api_exception", "error_message": error_message}

    
    print(f"[Failure] All retries have failed. Sample: {video_key}, Turn: {turn}")
    return {"error": "all_retries_failed", "error_message": "All failed"}


def process_video_multi_turn(video_key, video_folder, transcript, client, num_frames=16):

    print(f"\n{'=' * 80}")
    print(f"Processing video: {video_key}")
    print(f"Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    print(f"{'=' * 80}")


    turn_results = {}

    turn_descriptions = {
        1: "Objective Understanding ",
        2: "Hate Content Assumption ",
        3: "Non-Hate Content Assumption ",
        4: "Final Synthesis & Evaluation "
    }


    for turn in range(1, 5):
        print(f"\n{'-' * 60}")
        print(f"Turn {turn}/4: {turn_descriptions[turn]}")
        print(f"{'-' * 60}")


        raw_response = classify_multimodal_content_single_turn(
            video_key, transcript, client, video_folder, turn, turn_results, num_frames
        )


        if raw_response is None:

            turn_results[f'turn_{turn}'] = {
                'raw_response': None,
                'parsed_response': None,
                'parse_success': False,
                'error': 'No response received'
            }
            print(f"[Turn] No response received")
            continue


        if isinstance(raw_response, dict) and "error" in raw_response:
            error_type = raw_response.get("error")

            if error_type == "content_policy_violation":
                turn_results[f'turn_{turn}'] = {
                    'raw_response': "Content policy violation",
                    'parsed_response': None,
                    'parse_success': False,
                    'error': 'content_policy_violation'
                }
                print(f"[Turn] No response received")
                continue

            else:

                extracted_content = raw_response.get("extracted_content", "")
                raw_content = raw_response.get("raw_response", str(raw_response))

                print(f"\n📝 Model Response (Turn {turn}) - Error response, but content has been extracted:")
                print("🔹" + "🔹" * 25)
                print(f"Error type: {error_type}")
                if extracted_content:
                    print(f"Extract content: {extracted_content}")
                print(f"Original response: {raw_content}")
                print("🔹" + "🔹" * 25)


                if extracted_content:
                    parsed_response, success = clean_json_response(str(extracted_content))
                else:
                    parsed_response, success = None, False

                turn_results[f'turn_{turn}'] = {
                    'raw_response': raw_content,
                    'extracted_content': extracted_content,
                    'parsed_response': parsed_response if success else None,
                    'parse_success': success,
                    'error': error_type,
                    'error_details': raw_response
                }
                continue


        if isinstance(raw_response, str):

            print(f"\n📝 Model Response (Turn {turn}):")
            print("🔹" + "🔹" * 25)
            print(raw_response)
            print("🔹" + "🔹" * 25)


            parsed_response, success = clean_json_response(raw_response)


            turn_results[f'turn_{turn}'] = {
                'raw_response': raw_response,
                'parsed_response': parsed_response if success else None,
                'parse_success': success
            }

        else:
            print(f"\n📝 Model Response (Turn {turn}) - Unexpected response type:")
            print("🔹" + "🔹" * 25)
            print(f"Response type: {type(raw_response)}")
            print(f"Response content: {raw_response}")
            print("🔹" + "🔹" * 25)

            turn_results[f'turn_{turn}'] = {
                'raw_response': str(raw_response),
                'parsed_response': None,
                'parse_success': False,
                'error': 'unexpected_response_type',
                'response_type': str(type(raw_response))
            }

    
    final_result = {
        'video_id': video_key,
        'status': 'success',
        'num_frames': len(get_frames_from_folder(video_folder, num_frames)),
        'transcript': transcript,
        'multi_turn_results': turn_results,


        'summary': {
            'objective_analysis': turn_results.get('turn_1', {}).get('parsed_response'),
            'hate_evidence': turn_results.get('turn_2', {}).get('parsed_response'),
            'non_hate_evidence': turn_results.get('turn_3', {}).get('parsed_response'),
            'final_decision': turn_results.get('turn_4', {}).get('parsed_response'),
        }
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


def process_all_videos(video_frames_dir: str, transcript_json: str, output_file: str, filter_type: str = "all",
                       filter_count: int = 500, filter_position: str = "first"):
    """
    Process all videos using multi-round dialogue analysis

    Args:
        video_frames_dir: Path to video frame folder
        transcript_json: Path to transcript JSON file
        output_file: Path to output file

    filter_type: Filter type (‘all’, ‘hate’, ‘non_hate’, “first_n”, ‘exclude_first_n’)
        filter_count: Specifies quantity when filter_type includes a count (default 500)
        filter_position: Compatibility parameter, deprecated
    """

    print("OpenAI API...")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # If no environment variable exists, please set your API Key directly here.
        api_key = "Please replace with your actual API Key"  
        print("Warning: Please set the OPENAI_API_KEY environment variable or configure the API Key directly within your code.")

    client = OpenAI(api_key=api_key)


    transcripts = load_audio_transcripts(transcript_json)


    existing_results = []
    completed_video_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            for result in existing_results:
                video_id = result.get('video_id')
                multi_turn_results = result.get('multi_turn_results', {})


                if (video_id and
                        'turn_1' in multi_turn_results and
                        'turn_2' in multi_turn_results and
                        'turn_3' in multi_turn_results and
                        'turn_4' in multi_turn_results):
                    completed_video_ids.add(video_id)

            print(f"Found {len(completed_video_ids)} videos already completed with 4 turns")

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

    for video_folder in tqdm(videos_to_process, desc=f"Processing videos with multi-turn analysis"):
        video_id = video_folder.name
        transcript = transcripts.get(video_id, "")

        try:
            result = process_video_multi_turn(video_id, str(video_folder), transcript, client)
            results.append(result)


            if len(results) % 5 == 0:
                _save_results(results, output_file)

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'video_id': video_id,
                'status': 'error',
                'error': str(e),
                'multi_turn_results': None
            })


    _save_results(results, output_file)
    print(f"Multi-turn processing completed. Results saved to {output_file}")
    print(f"Total videos in results: {len(results)}")
    print(f"Newly processed: {len(videos_to_process)}")
    print(f"Previously completed: {len(completed_video_ids)}")


def main():

    # Video frame folder path
    video_frames_dir = "///"

    # Path to JSON file containing transcribed text
    transcript_json = "///"

    # Output file path - Modified to a different filename to avoid conflicts
    output_file = "multi_turn_hate_hatemm_detection_results_gpt5mini.json"


    filter_type = "all"
    filter_count = 500  # This parameter is not used in all mode.


    if not os.path.exists(video_frames_dir):
        print(f"Video frames directory not found: {video_frames_dir}")
        return

    if not os.path.exists(transcript_json):
        print(f"Transcript JSON file not found: {transcript_json}")
        return

    print(f"Processing videos with configuration:")
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
