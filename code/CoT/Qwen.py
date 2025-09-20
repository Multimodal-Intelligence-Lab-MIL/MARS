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


class HateDetectionInference:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"):

        self.model_name = model_name

        self.device = torch.device("cuda:0")
        print(f"Using device: {self.device}")


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

        frame_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            frame_files.extend(Path(video_folder).glob(ext))

        if not frame_files:
            print(f"No image files found in {video_folder}")
            return []


        frame_files.sort(key=lambda x: x.name)


        total_frames = len(frame_files)
        if total_frames <= num_frames:
            selected_files = frame_files
        else:

            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]
            selected_files = [frame_files[i] for i in indices]


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

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded transcripts for {len(data)} videos")
            return data
        except Exception as e:
            print(f"Error loading audio transcripts: {e}")
            return {}


    def create_prompt(self, transcript: str = "") -> str:

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

    def inference(self, images: List[Image.Image], prompt: str) -> str:

        try:

            if not images:
                print("No images provided for inference")
                return "Error: No images available for analysis"


            torch.cuda.empty_cache()


            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]


            for img in images:
                messages[0]["content"].append({
                    "type": "image",
                    "image": img
                })


            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })


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


            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )


            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )


            result = output_text[0].strip() if output_text and len(output_text) > 0 else "Error: No response generated"

            print(f"Model Response: {result}")
            print("-" * 50)


            del inputs, generated_ids
            torch.cuda.empty_cache()

            return result

        except Exception as e:

            torch.cuda.empty_cache()
            print(f"Error during inference: {e}")
            return f"Error during inference: {str(e)}"

    def process_video(self, video_id: str, video_folder: str, transcript: str = "") -> Dict:

        print(f"Processing video: {video_id}")


        images = self.sample_frames(video_folder, num_frames=16)

        if not images:
            return {
                'video_id': video_id,
                'status': 'error',
                'error': 'No frames could be loaded',
                'response': None
            }


        prompt = self.create_prompt(transcript)


        print(f"Video: {video_id} | Frames: {len(images)} | Transcript: {'Yes' if transcript.strip() else 'No'}")


        response = self.inference(images, prompt)

        return {
            'video_id': video_id,
            'status': 'success',
            'num_frames': len(images),
            'transcript': transcript,
            'response': response
        }

    def process_all_videos(self, video_frames_dir: str, transcript_json: str, output_file: str):

        transcripts = self.load_audio_transcripts(transcript_json)


        video_folders = [d for d in Path(video_frames_dir).iterdir() if d.is_dir()]
        video_folders.sort()
        print(f"Found {len(video_folders)} video folders")


        existing_results = []
        completed_video_ids = set()

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                for result in existing_results:
                    if result.get("status") == "success":
                        completed_video_ids.add(result.get("video_id"))
                print(f"Found {len(completed_video_ids)} completed videos, will skip them.")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                existing_results = []


        videos_to_process = [vf for vf in video_folders if vf.name not in completed_video_ids]
        print(f"Need to process {len(videos_to_process)} videos (skipping {len(completed_video_ids)} completed)")

        results = existing_results.copy()

        for video_folder in tqdm(videos_to_process, desc="Processing videos"):
            video_id = video_folder.name
            transcript = transcripts.get(video_id, "")

            try:
                result = self.process_video(video_id, str(video_folder), transcript)
                results.append(result)


                if len(results) % 10 == 0:
                    self._save_results(results, output_file)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'status': 'error',
                    'error': str(e),
                    'response': None
                })


        self._save_results(results, output_file)
        print(f"Processing completed. Results saved to {output_file}")
        print(f"Total videos in results: {len(results)}")
        print(f"Newly processed: {len(videos_to_process)}")
        print(f"Previously completed: {len(completed_video_ids)}")


    def _save_results(self, results: List[Dict], output_file: str):

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Results saved to {output_file}, total {len(results)} entries")
        except Exception as e:
            print(f"Error saving results: {e}")


def is_gpu_free(gpu_ids):

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
        print(f"[ERROR] GPU check failed: {e}")
        return False


def main():

    GPU_ID = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    gpu_list = [gid.strip() for gid in GPU_ID.split(",")]


    if not is_gpu_free(gpu_list):
        print(f"[FATAL] The selected GPU ({GPU_ID}) is currently in use; the programme has terminated.")
        sys.exit(1)

    print(f"[INFO] GPU {GPU_ID} available, commencing programme execution...")

    parser = argparse.ArgumentParser(description="Video Hate Detection using Qwen2.5-VL")
    parser.add_argument(
        "--video_frames_dir",
        type=str,
        default="",
        help="Directory containing video frame folders"
    )
    parser.add_argument(
        "--transcript_json",
        type=str,
        default="",
        help="JSON file containing audio transcripts"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Output file for results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="Model name"
    )

    args = parser.parse_args()


    if not os.path.exists(args.video_frames_dir):
        print(f"Video frames directory not found: {args.video_frames_dir}")
        return

    if not os.path.exists(args.transcript_json):
        print(f"Transcript JSON file not found: {args.transcript_json}")
        return


    detector = HateDetectionInference(args.model_name)


    detector.process_all_videos(
        args.video_frames_dir,
        args.transcript_json,
        args.output_file
    )


if __name__ == "__main__":
    main()