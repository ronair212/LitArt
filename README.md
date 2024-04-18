# LitArt

Welcome to the LitArt repository! This project merges the beauty of literature with digital innovation to create immersive experiences for literature enthusiasts.

## Demo

https://github.com/ronair212/LitArt/assets/78248225/32e2aa62-3a99-4fb1-9085-cd5db2b0ac10

## Introduction

LitArt is a platform that combines literary analysis with interactive technology, offering tools and features that enhance the reader's and writer's experience. Our goal is to make literary works more accessible and engaging through modern technology, providing insightful analyses, interactive content, and a community for literature lovers.

## Motivation

This project was inspired by the need to bridge the gap between traditional literary appreciation and the digital age. We aim to transform the solitary act of reading into a more engaging and communal activity, helping users explore literature in new and exciting ways.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or above
- GPU (A100, V100-SXM2)

## Installation

To set up LitArt for development, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ronair212/LitArt

2. **Install all the dependencies in your environment ([Link](https://github.com/ronair212/LitArt/blob/main/INSTALL.md) to setup environment)**
   ```bash
   pip install -r requirements.txt

## Running the Project

To run LitArt on your local machine:

```bash
streamlit run app.py
```

This will create a port that you can open in your browser.

To forward your port to another machine.

```bash
ssh -L 5000:<YOUR-PORT> <YOUR-CLOUD-LOGIN>
```

This is to forward your port to https://localhost:5000

## Training Models
If you want to train on your data, we have created scripts for the summarization and image generation model.

Training Summarizer
```bash
sh script_ts.sh
```

The training script for encoder-decoder models has the following training arguments 
```bash
--model "<MODEL-NAME>" \
--tokenizer "<MODEL-TOKEN>" \
--trainpath "<TRAINING-DATA-PATH>" \
--testpath "<TESTING-DATA-PATH>" \
--valpath "<VALIDATION-DATA-PATH>" \
--batchsize 16 \
--chapterlength 1024 \
--summarylength 128 \
--num_epochs 10 \
--log_path "<PATH-TO-SAVE-LOGS>" \
--cache_dir "<CACHE-DIRECTORY>"
```

Training Generator
```bash
sh script_sd.sh
```

### Flags
Generator
```bash
  --gradient_accumulation_steps=3 \
  --mixed_precision="fp16" \
  --max_train_steps=250\
  --learning_rate=2e-06 \
  --rank=8 \
  --lora=True \
```

You can tweak the flags mentioned above to try different configurations for the stable diffusion model

## Libraries and Tools Used

- Streamlit - Web framework
- Transformer, diffuser, PEFT - Model fine-tuning and training
- Pytorch Lighting - Framework
