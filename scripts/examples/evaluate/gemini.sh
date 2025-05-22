export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

python3 -m roboannotatorx/eval/eval_roboxvqa \
    --pred_path prediction.json \
    --model_version gemini-2.0-flash \
    --output_dir ./workdir \
    --output_json gemini-2.0-flash-score.json \
    --num_tasks 3