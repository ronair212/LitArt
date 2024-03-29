export GENERATOR="CompVis/stable-diffusion-v1-4"
export LORA="/work/LitArt/adwait/capstone/trained_adapters/fantasy_1.1.0/"
export SUMMARIZER="Llama"
export INPUT="../utilities/sample_inputs/Llama_inp_2.txt"
export FILENAME="Dune_fantasy_adapter"


export MAX_NEW_TOKEN="50"
export DO_SAMPLE="TRUE"
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
    --top_p=$TOP_P 