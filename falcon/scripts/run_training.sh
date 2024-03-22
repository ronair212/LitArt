
#!/bin/bash

# Default values
default_epochs=1
default_batch_size=2
default_learning_rate=2e-4
default_r=16
default_lora_alpha=32
default_lora_dropout=0.05
default_quant_4bit="True"
default_quant_8bit="False"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --learning_rate) learning_rate="$2"; shift 2 ;;
        --r) r="$2"; shift 2 ;;
        --lora_alpha) lora_alpha="$2"; shift 2 ;;
        --lora_dropout) lora_dropout="$2"; shift 2 ;;
        --quant_4bit) quant_4bit="$2"; shift 2 ;;
        --quant_8bit) quant_8bit="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

export BATCH_SIZE=${batch_size:-$default_batch_size}
export EPOCHS=${epochs:-$default_epochs}
export LEARNING_RATE=${learning_rate:-$default_learning_rate}
export R=${r:-$default_r}
export LORA_ALPHA=${lora_alpha:-$default_lora_alpha}
export LORA_DROPOUT=${lora_dropout:-$default_lora_dropout}
export QUANT_4BIT=${quant_4bit:-$default_quant_4bit}
export QUANT_8BIT=${quant_8bit:-$default_quant_8bit}


python main.py
