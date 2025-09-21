# MARS

This repository accompanies the paper *“Training-Free and Interpretable Hateful Video Detection via Multi-Stage Adversarial Reasoning (MARS)”*.


## Datasets

* [HateMM](https://github.com/hate-alert/HateMM)
* [MultiHateClip (MHC)](https://github.com/Social-AI-Studio/MultiHateClip)

Both datasets can be obtained from their respective GitHub repositories.

## Data Preprocessing

1. Extract video frames using **ffmpeg** or **OpenCV**:
   * Sample at **1 frame per second** and save as frame image folders.
     
2. Transcribe video audio into text using the **Whisper Medium model**.

## Repository Structure

```
.
├── code/        # Inference code
├── result/      # The compressed package contains the inference results JSON file, five-fold dataset split, and performance evaluation script.
└── README.md    # Documentation
```

