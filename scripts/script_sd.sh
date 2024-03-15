
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="../data/"
# export OUTPUT_DIR="./trained_models/"

# accelerate launch train_text_to_image.py \
#   --mixed_precision="fp16" \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a <book cover> for a magical fantasy book" \
#   --train_text_encoder \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=3 \
#   --learning_rate=2e-6 \
#   --lr_scheduler="DDIM" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=400 \
#   --gradient_checkpointing \
#   --use_8bit_adam \

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="../data/"
export OUTPUT_DIR="/work/LitArt/adwait/capstone/trained_models/model_1.1.0/"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=1500 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --use_8bit_adam \
