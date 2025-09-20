# MARS

This repository accompanies the paper *“Training-Free and Interpretable Hateful Video Detection via Multi-Stage Adversarial Reasoning (MARS)”*.


## Datasets

* [HateMM](https://github.com/hate-speech-datasets/HateMM)
* [MultiHateClip (MHC)](https://github.com/hate-speech-datasets/MultiHateClip)

Both datasets can be obtained from their respective GitHub repositories.

## Data Preprocessing

1. Extract video frames using **ffmpeg** or **OpenCV**:
   * Sample at **1 frame per second** and save as frame image folders.
     
2. Transcribe video audio into text using the **Whisper Medium model**.

## Repository Structure

```
.
├── code/        # Inference code
├── result/      # Inference results (compressed), dataset splits, and performance evaluation scripts
└── README.md    # Documentation
```

