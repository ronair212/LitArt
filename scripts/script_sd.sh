
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="../data/"
export OUTPUT_DIR="./trained_models/"

accelerate launch train_text_to_image.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a <book cover> for a magical fantasy book" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=3 \
  --learning_rate=2e-6 \
  --lr_scheduler="DDIM" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --gradient_checkpointing \
  --use_8bit_adam \
