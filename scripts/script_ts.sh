# python train_text.py --model "google-t5/t5-small" --tokenizer "google-t5/t5-small" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 512 --summarylength 64 --num_epochs 1 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

# python train_text.py --model "google-t5/t5-base" --tokenizer "google-t5/t5-base" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 1024 --summarylength 128 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

# python train_text.py --model "google-t5/t5-large" --tokenizer "google-t5/t5-large" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 64 --chapterlength 1024 --summarylength 128 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  '"work/LitArt/cache"

# python train_text.py --model "google/pegasus-xsum" --tokenizer "google/pegasus-xsum" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 512 --summarylength 64 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

# python train_text.py --model "google/pegasus-large" --tokenizer "google/pegasus-large" --trainpath "../Datasets/Training_data.csv" --testpath "../Datasets/Testing_data.csv" --valpath "../Datasets/Validation_data.csv" --batchsize 16 --chapterlength 1024 --summarylength 128 --num_epochs 10 --log_path "/work/LitArt/verma/" --cache_dir  "/work/LitArt/cache"

python train_text.py \
--model "google/pegasus-xsum" \
--tokenizer "google/pegasus-xsum" \
--trainpath "/work/LitArt/data/generated_summaries/train_dataset_with_summaries.csv" \
--testpath "/work/LitArt/data/generated_summaries/test_dataset_with_summaries.csv" \
--valpath "/work/LitArt/data/generated_summaries/validation_dataset_with_summaries.csv" \
--batchsize 32 \
--chapterlength 512 \
--summarylength 64 \
--num_epochs 10 \
--log_path "/work/LitArt/verma/" \
--cache_dir  "/work/LitArt/cache" 