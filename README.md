MLSP 2025 BIT Baseline
This repository contains the baseline PyTorch code for the BIT (Body In Transit) competition at the MLSP 2025 congress in Turkey. Participants can use this code to jump-start their work on the classification challenge and generate a valid submission for the leaderboard.

Overview
The provided code includes:

train.py: Trains a simple baseline model.
test.py: Generates the submission file in the required format.
Participants can run these scripts sequentially to recreate the baseline entry.

Competition Tracks
Classification Challenge
In the classification challenge, participants must predict whether a user feels lighter or heavier compared to the control condition. To facilitate this:

The original dataset contains labels on a Likert scale from 1 to 7 (ranging from light to heavy).
For the competition, these are transformed into three categories:
-1: User feels lighter than the control.
0: Control condition.
1: User feels heavier than the control.
The dataset class in the code handles this label transformation. See the Label Transformation section for details.

Beyond the Classification Challenge
Participants are also invited to submit novel contributions that extend beyond the standard classification task. This track encourages innovative approaches using any method the authors see fit.

For more details on both submission types, please visit the competition website.

Installation
Clone the Repository:

bash
Copiar
Editar
git clone https://github.com/tmcortes/MLSP_25_bit__baseline.git
cd MLSP_25_bit__baseline
Create and Activate a Virtual Environment (optional but recommended):

bash
Copiar
Editar
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install Required Packages:

The code relies on PyTorch and several other libraries. Install the required packages using:

bash
Copiar
Editar
pip install -r requirements.txt
(If a requirements.txt file is not provided, please ensure that PyTorch and any other dependencies mentioned in the code are installed.)

Usage
Training:

Run the training script to train the baseline model:

bash
Copiar
Editar
python train.py
Testing:

After training, generate your submission by running:

bash
Copiar
Editar
python test.py
Refer to the comments within the code for further details on configuration and parameters.

Label Transformation
The original labels in the dataset come as a Likert scale (1 to 7, representing light to heavy). For the classification competition, the goal is to determine whether the user feels lighter or heavier compared to a control condition. The transformation process is as follows:

python
Copiar
Editar
# the original labels come in a likert scale from 1 to 7 (light to heavy)
# for the classification competition we are interested in whether the user
# feels lighter or heavier than with the control. Thus: we are going to 
# re-label those 1-7 likert scale indexes, to -1 (the user feels lighter
# than with the control audio condition) 0 (is a control condition) and (1)
# the subject claims to feel heavier than with the control condition).
#
# Since for the control walks, the subject can give different 1-7 labels, 
# we get the median of the control labels to use it as a reference. Then,
# we label each walk as -1 if the user claims a 1-7 index smaller than the 
# control median, and +1 if the user claims a 1-7 index larger than the control 
# median. 
# 
# Our models need to learn to predict given a sample of a condition (LF, HF or control)
# and a sample of the control condition (X, Xc), if the user feels lighter, heavier or the same
# as with the control condition.
For more details, please refer to the inline comments within the dataset class in the code.

Related Research
For a comprehensive explanation of the data acquisition process, please see the following paper:
Data Acquisition Paper

Contact
For any questions or issues, please open an issue in this repository or contact the repository maintainer.

Enjoy competing and best of luck in the challenge!
