export GENERATOR="CompVis/stable-diffusion-v1-4"
export LORA="/work/LitArt/adwait/capstone/trained_adapters/suspense_1.1.0/"
export SUMMARIZER="Llama"
export INPUT="../utilities/sample_inputs/Gangs_of_wassepur.txt"
export FILENAME="Gangs_of_wassepur_suspense_adapter"


export MAX_NEW_TOKEN="500"
export DO_SAMPLE="False"
export TEMPERATURE="1.0"
export TOP_P="0.8"


python inference.py \
    -c=$INPUT \
    -s=$SUMMARIZER \
    -g=$GENERATOR \
    -l=$LORA \
    -f=$FILENAME \
    -max_new_tokens=$MAX_NEW_TOKEN \
    -do_sample=$DO_SAMPLE \
    -temperature=$TEMPERATURE \
    -top_p=$TOP_P \