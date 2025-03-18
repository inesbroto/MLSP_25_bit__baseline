 # BodyInTransit Data Competition at MLSP@2025 - Baseline Pytorch Implementation
 
 **Repository for the BiT Data Challenge**  
 **Competition Website**: [https:bodyintransit.eu/bit-data-competition/](https:bodyintransit.eu/bit-data-competition/)
 
 ---
 
 ## Overview
 This repository provides a PyTorch baseline implementation for participants of the MLSP 2025 "Body in Transit" (BIT) Data Competition. The code includes:
 - Data loading
 - A simple model architecture for the baseline competition result
 - Training (`train.py`) and inference (`test.py`) pipelines
 - Submission formatting utilities for the competition
 
 Participants can use this baseline to jumpstart their solutions for the **classification challenge** or the **"Beyond Classification" contribution track**.
 
 ---
 
 ## Getting Started
 
 ### Prerequisites
 - Python 3.9.18
 
 ### Installation
 1. Clone the repository:
    ```bash
    git clone https:github.com/[your-username]/bit-mlsp2025-baseline.git
    cd bit-mlsp2025-baseline
    ```
 2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 
 ### Run Baseline
 1. Train the model:
    ```bash
    python train.py --db_folder ../dataset/MLSP_bit_dataset/train
    ```
 2. Generate submissions:
    ```bash
    python test.py --test_path ../dataset/MLSP_bit_dataset/test --model ./checkpoint_epoch_10.pth
    ```
 
 ---
 
 ## Dataset
 - **Original Labels**: Likert scale scores (1-7) as described in the [BIT Data Acquisition Paper](https:dl.acm.org/doi/10.1145/3613904.3642651)
 - **Competition Labels**: Transformed to {-1, 0, +1} where:
   - **-1**: Perceived lighter than with control audio
   - **0**: No perceived difference with respect to control audio
   - **+1**: Perceived heavier than control audio
 
 The included `SubjectDataset` class handles this transformation automatically (see code comments for implementation details).
 
 ---
 
 ## Competition Tracks
 ### 1. Classification Challenge
 - Predict user perception relative to control conditions
 - Uses transformed ternary labels {-1, 0, +1}
 - Submissions evaluated on macro F1-Score (check submissionScore.py and test.py for details)
 
 ### 2. Beyond Classification Contribution
 - Open-ended track for novel methodological contributions
 - May use raw Likert scores (1-7) or alternative approaches
 - Submissions evaluated on technical novelty and impact
 
 **Both tracks** accept paper submissions to MLSP 2025. Participants may use this repository as a starting point for either track.
 
 ---
 
 ## Submission for the classification competition
 1. Example code for format predictions in `test.py` and `SubjectDataset.generateTestSubmission()`
 2. Follow guidelines on the [competition website](https:bodyintransit.eu/bit-data-competition/)
 
 
 ---
 
 ## Contact
 For competition inquiries: [tmcortes@ing.uc3m.es](mailto:tmcortes@ing.uc3m.es)
 
 ---
 
 **Good luck to all participants!** 🚀
