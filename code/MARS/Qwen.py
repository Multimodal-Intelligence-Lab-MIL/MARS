#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import subprocess
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from pathlib import Path
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm


def process_vision_info(messages):
    """Process visual information within messages, extract images and videos"""
    image_inputs = []
    video_inputs = []

    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for content in message["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
                elif content["type"] == "video":
                    video_inputs.append(content["video"])

    return image_inputs, video_inputs


class MultiTurnHateDetection:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"):
        """
        Initialise the model and processor

        Args:
            model_name: Model name
        """
        self.model_name = model_name
        self.device = torch.device("cuda:0")
        print(f"Using device: {self.device}")

        # Configure int8 quantisation
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )

        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loading the model and processor"""
        try:
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            print("Loading model with int8 quantization...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            print("Model loaded successfully!")
            print(f"Model device: {next(self.model.parameters()).device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def sample_frames(self, video_folder: str, num_frames: int = 16) -> List[Image.Image]:
        """Sample a specified number of frames uniformly from the video folder"""
        frame_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            frame_files.extend(Path(video_folder).glob(ext))

        if not frame_files:
            print(f"No image files found in {video_folder}")
            return []

        # Sort by filename
        frame_files.sort(key=lambda x: x.name)

        # uniform sampling
        total_frames = len(frame_files)
        if total_frames <= num_frames:
            selected_files = frame_files
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]
            selected_files = [frame_files[i] for i in indices]

        # Load image
        images = []
        for file_path in selected_files:
            try:
                img = Image.open(file_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Failed to load image {file_path}: {e}")

        print(f"Sampled {len(images)} frames from {video_folder}")
        return images

    def load_audio_transcripts(self, json_path: str) -> Dict[str, str]:
        """Loading audio transcription text"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded transcripts for {len(data)} videos")
            return data
        except Exception as e:
            print(f"Error loading audio transcripts: {e}")
            return {}

    def create_turn_prompt(self, turn: int, transcript: str = "", previous_results: Dict = None) -> str:
        """
        Generate prompts for different rounds

        Args:
            turn: Round number (1-4)
            transcript: Transcript text
            previous_results: Results from previous rounds

        Returns:
            Prompt for the specified round
        """

        if turn == 1:
            # First round: Purely objective comprehension
            base_prompt = """You are a content analysis expert. Provide a detailed, objective description of this video content.

Task: Describe what you observe in the video frames and transcript:
1. objective_visual_description: Describe what you see - objects, people, actions, settings, text, symbols, colors, expressions, gestures, etc.

Rules:
- Be comprehensive and detailed
- NO analysis, interpretations, or value judgments
- Simply report what you observe
- Return ONLY valid JSON with one key: ["objective_visual_description"]"""

        elif turn == 2:
            # Second Round: Hate Speech Hypothesis
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
            # Third round: Non-hate content hypothesis
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
            # Fourth Round: Comprehensive Assessment
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

        # Add transcription text
        if transcript.strip():
            prompt = base_prompt + f"\n\nTranscript: {transcript}"
        else:
            prompt = base_prompt + "\n\nTranscript: No transcript available."

        return prompt

    def inference_single_turn(self, images: List[Image.Image], prompt: str) -> str:
        """Execute single-round reasoning"""
        try:
            if not images:
                print("No images provided for inference")
                return "Error: No images available for analysis"

            # Clear the GPU cache
            torch.cuda.empty_cache()

            # Create messages
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]

            # Add all images
            for img in images:
                messages[0]["content"].append({
                    "type": "image",
                    "image": img
                })

            # 添加文本prompt
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })

            # Processing input
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(self.device)

            # Generate a response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            result = output_text[0].strip() if output_text and len(output_text) > 0 else "Error: No response generated"

            # Clear the GPU cache
            del inputs, generated_ids
            torch.cuda.empty_cache()

            return result

        except Exception as e:
            torch.cuda.empty_cache()
            print(f"Error during inference: {e}")
            return f"Error during inference: {str(e)}"

    def clean_json_response(self, response_text: str) -> tuple:
        """Clean up model outputs and extract JSON content"""
        try:
            cleaned = response_text.strip()

            # Remove Markdown markup
            if cleaned.startswith('```json') and cleaned.endswith('```'):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith('```') and cleaned.endswith('```'):
                cleaned = cleaned[3:-3].strip()

            # Parsing JSON
            parsed_json = json.loads(cleaned)
            return parsed_json, True

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return response_text, False

    def process_video_multi_turn(self, video_id: str, video_folder: str, transcript: str = "") -> Dict:
        """
        Multi-round dialogue analysis for individual videos

        Args:
            video_id: Video ID
            video_folder: Video frame folder path
            transcript: Transcribed text

        Returns:
            Dictionary containing results for all rounds
        """
        print(f"\n{'=' * 80}")
        print(f"Processing video: {video_id}")
        print(f"Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
        print(f"{'=' * 80}")

        # Sampling frame
        images = self.sample_frames(video_folder, num_frames=16)
        if not images:
            return {
                'video_id': video_id,
                'status': 'error',
                'error': 'No frames could be loaded',
                'multi_turn_results': None
            }

        # Store the results of each round
        turn_results = {}

        # Definition of Round Description
        turn_descriptions = {
            1: "Objective Understanding",
            2: "Hate Content Assumption",
            3: "Non-Hate Content Assumption",
            4: "Final Synthesis & Evaluation"
        }

        # Conduct four rounds of dialogue
        for turn in range(1, 5):
            print(f"\n{'-' * 60}")
            print(f"Turn {turn}/4: {turn_descriptions[turn]}")
            print(f"{'-' * 60}")

            # Create the prompt for the current round
            prompt = self.create_turn_prompt(turn, transcript, turn_results)

            # Executing reasoning
            raw_response = self.inference_single_turn(images, prompt)

            # Real-time printing of model responses
            print(f"\n📝 Model Response (Turn {turn}):")
            print("🔹" + "🔹" * 25)
            print(raw_response)
            print("🔹" + "🔹" * 25)

            # Cleaning and Parsing Responses
            parsed_response, success = self.clean_json_response(raw_response)

            # Store the results
            turn_results[f'turn_{turn}'] = {
                'prompt': prompt,
                'raw_response': raw_response,
                'parsed_response': parsed_response if success else None,
                'parse_success': success
            }

        # Construct the final result
        final_result = {
            'video_id': video_id,
            'status': 'success',
            'num_frames': len(images),
            'transcript': transcript,
            'multi_turn_results': turn_results,

            # Extract key results for quick access
            'summary': {
                'objective_analysis': turn_results.get('turn_1', {}).get('parsed_response'),
                'hate_evidence': turn_results.get('turn_2', {}).get('parsed_response'),
                'non_hate_evidence': turn_results.get('turn_3', {}).get('parsed_response'),
                'final_evaluation': turn_results.get('turn_4', {}).get('parsed_response')
            }
        }

        return final_result

    def process_all_videos(self, video_frames_dir: str, transcript_json: str, output_file: str):
        """Process all videos: Skip any video_id already present in the existing results file."""

        transcripts = self.load_audio_transcripts(transcript_json)


        existing_results = []
        existing_ids = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                for r in existing_results:
                    vid = r.get('video_id')
                    if isinstance(vid, str) and vid:
                        existing_ids.add(vid)
                print(f"Found {len(existing_ids)} existing video_id(s) in {output_file}")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                existing_results = []
                existing_ids = set()


        video_folders = [d for d in Path(video_frames_dir).iterdir() if d.is_dir()]
        video_folders.sort()
        print(f"Found {len(video_folders)} video folders")


        videos_to_process = [vf for vf in video_folders if vf.name not in existing_ids]
        print(f"Need to process {len(videos_to_process)} videos (skipping {len(existing_ids)} existing)")

        results = existing_results.copy()

        for video_folder in tqdm(videos_to_process, desc="Processing videos with multi-turn analysis"):
            video_id = video_folder.name
            transcript = transcripts.get(video_id, "")

            try:
                result = self.process_video_multi_turn(video_id, str(video_folder), transcript)
                results.append(result)


                if len(results) % 5 == 0:
                    self._save_results(results, output_file)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'status': 'error',
                    'error': str(e),
                    'multi_turn_results': None
                })


        self._save_results(results, output_file)
        print(f"Multi-turn processing completed. Results saved to {output_file}")
        print(f"Total videos listed: {len(video_folders)}")
        print(f"Processed this run: {len(videos_to_process)}")
        print(f"Skipped (already existed): {len(existing_ids)}")
    '''
    def process_all_videos(self, video_frames_dir: str, transcript_json: str, output_file: str):
        """Process all videos to ensure each undergoes four rounds of dialogue analysis."""
        
        transcripts = self.load_audio_transcripts(transcript_json)

        
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

        
        videos_to_process = [vf for vf in video_folders if vf.name not in completed_video_ids]
        print(f"Need to process {len(videos_to_process)} videos (skipping {len(completed_video_ids)} completed)")

        results = existing_results.copy()  

        for video_folder in tqdm(videos_to_process, desc="Processing videos with multi-turn analysis"):
            video_id = video_folder.name
            transcript = transcripts.get(video_id, "")

            try:
                result = self.process_video_multi_turn(video_id, str(video_folder), transcript)
                results.append(result)

                
                if len(results) % 5 == 0:  
                    self._save_results(results, output_file)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'status': 'error',
                    'error': str(e),
                    'multi_turn_results': None
                })

        
        self._save_results(results, output_file)
        print(f"Multi-turn processing completed. Results saved to {output_file}")
        print(f"Total videos processed: {len(results)}")
        print(f"Newly processed: {len(videos_to_process)}")
        print(f"Previously completed: {len(completed_video_ids)}")
    '''
    '''    
    def process_all_videos(self, video_frames_dir: str, transcript_json: str, output_file: str):
        """Process all videos"""
        
        transcripts = self.load_audio_transcripts(transcript_json)

        
        video_folders = [d for d in Path(video_frames_dir).iterdir() if d.is_dir()]
        video_folders.sort()

        print(f"Found {len(video_folders)} video folders")

        results = []

        for video_folder in tqdm(video_folders, desc="Processing videos with multi-turn analysis"):
            video_id = video_folder.name
            transcript = transcripts.get(video_id, "")

            try:
                result = self.process_video_multi_turn(video_id, str(video_folder), transcript)
                results.append(result)

                
                if len(results) % 5 == 0:  
                    self._save_results(results, output_file)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'status': 'error',
                    'error': str(e),
                    'multi_turn_results': None
                })

        
        self._save_results(results, output_file)
        print(f"Multi-turn processing completed. Results saved to {output_file}")'''

    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving results: {e}")


def is_gpu_free(gpu_ids):
    """Check whether the specified GPU is idle"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits']
        ).decode()
        for line in result.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                gpu_index = parts[0].strip()
                utilization = int(parts[1].strip())
                if gpu_index in gpu_ids and utilization > 5:
                    return False
        return True
    except Exception as e:
        print(f"[ERROR] GPU inspection failed: {e}")
        return False


def main():
    # GPU设置
    GPU_ID = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    gpu_list = [gid.strip() for gid in GPU_ID.split(",")]

    # 检查GPU
    if not is_gpu_free(gpu_list):
        print(f"[FATAL] GPU（{GPU_ID}）currently occupied; programme terminated.")
        sys.exit(1)

    print(f"[INFO] GPU {GPU_ID} available, commencing multi-round dialogue programme...")

    parser = argparse.ArgumentParser(description="Multi-turn Video Hate Detection using Qwen2.5-VL")
    parser.add_argument(
        "--video_frames_dir",
        type=str,
        default="your video frames path",
        help="Directory containing video frame folders"
    )
    parser.add_argument(
        "--transcript_json",
        type=str,
        default="your transcripts json file path",
        help="JSON file containing audio transcripts"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output json file",
        help="Output file for results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="Model name"
    )

    args = parser.parse_args()

    # Check the path
    if not os.path.exists(args.video_frames_dir):
        print(f"Video frames directory not found: {args.video_frames_dir}")
        return

    if not os.path.exists(args.transcript_json):
        print(f"Transcript JSON file not found: {args.transcript_json}")
        return

    # Initialise the multi-turn dialogue detector
    detector = MultiTurnHateDetection(args.model_name)

    # Process all videos
    detector.process_all_videos(
        args.video_frames_dir,
        args.transcript_json,
        args.output_file
    )


if __name__ == "__main__":
    main()