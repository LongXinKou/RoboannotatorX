export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

python3 -m roboannotatorx/eval/eval_roboxvqa \
    --pred_path prediction.json \
    --model_version gpt-4o \
    --output_dir ./workdir \
    --output_json gpt-4o-score.json \
    --num_tasks 3