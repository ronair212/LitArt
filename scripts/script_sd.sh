
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/work/LitArt/adwait/capstone/data/speculative_fiction_dataset"
export OUTPUT_DIR="/work/LitArt/adwait/capstone/trained_adapters/speculative_fiction_1.1.0/"
export DATA_DIR="/work/LitArt/data/images_segregated/resolution_100/Speculative_fiction.csv"

accelerate launch train_generator.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --original_data_path=$DATA_DIR \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=3 \
  --mixed_precision="fp16" \
  --max_train_steps=300\
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --use_8bit_adam \
  --rank=3 \
  --lora=True \
