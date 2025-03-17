 # BIT Data Competition 2025 - Baseline Implementation
 
 **Repository for the MLSP Congress 2025 Classification Challenge**  
 **Competition Website**: [https:bodyintransit.eu/bit-data-competition/](https:bodyintransit.eu/bit-data-competition/)
 
 ---
 
 ## Overview
 This repository provides a PyTorch baseline implementation for participants of the MLSP 2025 "Body in Transit" (BIT) Data Competition. The code includes:
 - Data loading and preprocessing scripts
 - A simple model architecture template
 - Training (`train.py`) and inference (`test.py`) pipelines
 - Submission formatting utilities for the competition
 
 Participants can use this baseline to jumpstart their solutions for the **classification challenge** or the **"Beyond Classification" contribution track**.
 
 ---
 
 ## Getting Started
 
 ### Prerequisites
 - Python 3.8+
 - PyTorch 2.0+
 - Additional dependencies: `numpy`, `pandas`, `tqdm`
 
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
    python train.py --data_path /path/to/dataset
    ```
 2. Generate submissions:
    ```bash
    python test.py --data_path /path/to/test_data --model_checkpoint ./model.pth
    ```
 
 ---
 
 ## Dataset
 - **Original Labels**: Likert scale scores (1-7) as described in the [BIT Data Acquisition Paper](https:dl.acm.org/doi/10.1145/3613904.3642651)
 - **Competition Labels**: Transformed to {-1, 0, +1} where:
   - **-1**: Perceived lighter than control
   - **0**: No perceived difference
   - **+1**: Perceived heavier than control
 
 The included `BITDataset` class handles this transformation automatically (see code comments for implementation details).
 
 ---
 
 ## Competition Tracks
 ### 1. Classification Challenge
 - Predict user perception relative to control conditions
 - Uses transformed ternary labels {-1, 0, +1}
 - Submissions evaluated on classification accuracy
 
 ### 2. Beyond Classification Contribution
 - Open-ended track for novel methodological contributions
 - May use raw Likert scores (1-7) or alternative approaches
 - Submissions evaluated on technical novelty and impact
 
 **Both tracks** accept paper submissions to MLSP 2025. Participants may use this repository as a starting point for either track.
 
 ---
 
 ## Submission
 1. Format predictions using `prepare_submission.py`
 2. Follow guidelines on the [competition website](https:bodyintransit.eu/bit-data-competition/)
 3. Submit both competition entry and paper draft by the deadline
 
 ---
 
 ## Repository Structure
 ```
 .
 ├── data/               # Example data structure (placeholder)
 ├── src/
 │   ├── dataset.py      # BITDataset class with label transformation
 │   ├── model.py        # Baseline model architecture
 │   ├── train.py        # Training script
 │   ├── test.py         # Inference script
 │   └── utils/          # Submission formatting helpers
 ├── requirements.txt
 └── README.md
 ```
 
 ---
 
 ## Citation
 If you use this dataset or code, please cite:  
 ```bibtex
 @inproceedings{bit2024,
   title={Body in Transit: Multimodal Perception Dataset},
   author={Competition Organizers},
   booktitle={Proceedings of ACM Multimedia},
   year={2024},
   doi={10.1145/3613904.3642651}
 }
 ```
 
 ---
 
 ## Contact
 For competition inquiries: [competition-contact-email@bodyintransit.eu](mailto:competition-contact-email@bodyintransit.eu)
 
 ---
 
 **Good luck to all participants!** 🚀
