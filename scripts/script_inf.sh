export GENERATOR="CompVis/stable-diffusion-v1-4"
export LORA="/work/LitArt/adwait/capstone/trained_adapters/suspense_1.1.0/"
export SUMMARIZER="Llama"
export INPUT="../utilities/sample_inputs/Gangs_of_wassepur.txt"
export FILENAME="Gangs_of_wassepur_suspense_adapter"

python inference.py \
    -c=$INPUT \
    -s=$SUMMARIZER \
    -g=$GENERATOR \
    -l=$LORA \
    -f=$FILENAME 