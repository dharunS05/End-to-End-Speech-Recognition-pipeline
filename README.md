# End-to-End Speech Recognition Pipeline

## Overview

This repository provides a modular and extensible **Hybrid HMM-RNN Speech-to-Text Pipeline** for automatic speech recognition (ASR) research and applications. The pipeline integrates advanced audio preprocessing, robust acoustic modeling, and flexible decoding strategies.

## Key Features

- **Audio Preprocessing:** Voice Activity Detection (VAD), noise reduction, normalization, and MFCC extraction using Librosa.
- **Acoustic Modeling:** LSTM-based RNN acoustic model implemented in PyTorch.
- **Decoding:** Sequence-to-sequence (Seq2Seq) decoder with optional KenLM-based language model for beam search decoding.
- **Evaluation:** Automated computation of Word Error Rate (WER).
- **Inference:** Simple interface for transcribing audio files.
- **Dataset:** Supports the [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) for training and evaluation.

## Project Structure

- `main.py`: End-to-end training and inference pipeline.
- `dataset/`: Contains the LJSpeech dataset (`wavs/` and `metadata.csv`).

## Getting Started

### 1. Prepare the Dataset

Download the [LJSpeech-1.1 dataset](https://keithito.com/LJ-Speech-Dataset/) and extract it to:

```
/content/dataset/LJSpeech-1.1/
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install torch numpy librosa tqdm
# Optional: For language model decoding
pip install kenlm
```

### 3. Training and Inference

To train the model and run inference:

```bash
python main.py
```

To transcribe a specific audio file:

```python
hypothesis = inference("path/to/audio.wav", input_dim=13, hidden_dim=128, output_dim=256, use_kenlm=False)
print(hypothesis)
```

## Requirements

- Python 3.7+
- torch
- numpy
- librosa
- tqdm
- kenlm (optional, for language model decoding)

## Acknowledgements

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Librosa](https://librosa.org/)
- [KenLM](https://github.com/kpu/kenlm)
- PyTorch

---

For questions, suggestions, or contributions, please open an issue or submit a pull request.