export GENERATOR="CompVis/stable-diffusion-v1-4"
export LORA="/work/LitArt/adwait/capstone/trained_adapters/fiction_3.0.1/"
export SUMMARIZER="Llama"
export INPUT="../utilities/sample_inputs/lady_and_library.txt"
export FILENAME="lady_and_library_fiction_adapter_3"
export INFERENCE_STEPS=100


export MAX_NEW_TOKEN="500"
export DO_SAMPLE="FALSE"
export TEMPERATURE="1"
export TOP_P="0.8"


python inference.py \
    -c=$INPUT \
    -s=$SUMMARIZER \
    -g=$GENERATOR \
    -l=$LORA \
    -f=$FILENAME \
    --max_new_tokens=$MAX_NEW_TOKEN \
    --do_sample=$DO_SAMPLE \
    --temperature=$TEMPERATURE \
    --top_p=$TOP_P \
    -i=$INFERENCE_STEPS