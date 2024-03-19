# python train_text.py --model "google-t5/t5-small" --tokenizer "google-t5/t5-small" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 512 --summarylength 64 --num_epochs 1 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

# python train_text.py --model "google-t5/t5-base" --tokenizer "google-t5/t5-base" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 1024 --summarylength 128 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

# python train_text.py --model "google-t5/t5-large" --tokenizer "google-t5/t5-large" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 64 --chapterlength 1024 --summarylength 128 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  '"work/LitArt/cache"

# python train_text.py --model "google/pegasus-xsum" --tokenizer "google/pegasus-xsum" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 512 --summarylength 64 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

# python train_text.py --model "google/pegasus-large" --tokenizer "google/pegasus-large" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 1024 --summarylength 128 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

python train_text_causal.py \
--model "tiiuae/falcon-7b" \
--tokenizer "tiiuae/falcon-7b" \
--trainpath "/work/LitArt/data/chunked_dataset/train_dataset_with_summaries.csv" \
--testpath "/work/LitArt/data/chunked_dataset/test_dataset_with_summaries.csv" \
--valpath "/work/LitArt/data/chunked_dataset/validation_dataset_with_summaries.csv" \
--batchsize 16 \
--chapterlength 1024 \
--summarylength 128 \
--num_epochs 10 \
--log_path "/work/LitArt/verma/" \
--cache_dir  "/work/LitArt/cache" 


# python train_text.py \
# --model "facebook/bart-large" \
# --tokenizer "facebook/bart-large" \
# --trainpath "/work/LitArt/data/chunked_dataset/train_dataset_with_summaries.csv" \
# --testpath "/work/LitArt/data/chunked_dataset/test_dataset_with_summaries.csv" \
# --valpath "/work/LitArt/data/chunked_dataset/validation_dataset_with_summaries.csv" \
# --batchsize 16 \
# --chapterlength 1024 \
# --summarylength 128 \
# --num_epochs 10 \
# --log_path "/work/LitArt/verma/" \
# --cache_dir  "/work/LitArt/cache" 