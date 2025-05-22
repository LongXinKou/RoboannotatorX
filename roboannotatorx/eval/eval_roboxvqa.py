from openai import OpenAI
import os
import argparse
import json
import re
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-LLM")
    parser.add_argument("--model_version", help="Evaluation LLM")
    parser.add_argument("--pred_path", help="The path to file containing prediction.")
    parser.add_argument("--output_dir", help="The path to save annotation json files.")
    parser.add_argument("--output_json", help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    args = parser.parse_args()
    return args


def extract_dict_from_text(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except Exception as e:
            print("解析失败:", e)
            return None
    return None


def annotate(prediction_set, caption_files, output_dir, model_version):
    for file in tqdm(caption_files):
        key = file[:-5]  # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred'][0] if isinstance(qa_set['pred'], list) else qa_set['pred']
        try:
            # Compute the correctness score
            if pred != "":
                completion = client.chat.completions.create(
                    model=model_version,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "Your task is to compare the Predicted Answer with the Correct Answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. Consider the following guidelines:"
                                "1. Minor differences in object descriptions should not necessarily result in a negative evaluation if the overall action is correct."
                                "2. Consider the potential for slight misinterpretations in visual details, especially for similar objects or surfaces.\n"
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
                response_message = completion.choices[0].message.content
                response_dict = extract_dict_from_text(response_message)
            else:
                response_dict = {"pred": "no", "score": 0}
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def get_name_from_path(model_path, index=-1):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    return model_paths[index]


def main():
    # Parse arguments.
    args = parse_args()
    model_name = get_name_from_path(args.pred_path, index=-1).split('.')[0]
    dataset_name = get_name_from_path(args.pred_path, index=-2)
    output_dir = os.path.join(args.output_dir, dataset_name, model_name, args.model_version)
    json_path = os.path.join(output_dir, args.output_json)

    if not os.path.exists(json_path):
        if args.num_chunks > 1:
            pred_contents = []
            for _idx in range(args.num_chunks):
                file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}.json")
                pred_contents += [json.loads(line) for line in open(file)]

        else:
            pred_contents = [json.loads(line) for line in open(args.pred_path)]

        # Dictionary to store the count of occurrences for each video_id
        video_id_counts = {}
        new_pred_contents = []

        # Iterate through each sample in pred_contents
        for sample in pred_contents:
            video_id = sample['id'] if '/' not in sample['id'] else sample['id'].replace('/', '_')
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            # Create a new sample with the modified key
            new_sample = sample
            new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        # Generating list of id's and corresponding files
        id_list = [x['id'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        # Generate output directory if not exists.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Preparing dictionary of question-answer sets
        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['id']
            question = sample['question']
            answer = sample['answer']
            pred = sample['pred']
            question_type = sample['question_type']
            qa_set = {"q": question, "a": answer, "pred": pred, "question_type": question_type}
            prediction_set[id] = qa_set

        num_tasks = args.num_tasks

        # While loop to ensure that all captions are processed.
        while True:
            try:
                # Files that have not been processed yet.
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                # Files that have not been processed yet.
                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                # Break the loop when there are no incomplete files
                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                # Split tasks into parts.
                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, output_dir, args.model_version) for part in all_parts]

                # Use a pool of workers to process the files in parallel.
                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e:
                print(f"Error: {e}")

        # Combine all the processed files into one
        combined_contents = {}

        # Iterate through json files
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content
                os.remove(file_path)

        # Write combined content to a json file
        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")
    else:
        with open(json_path, "r") as json_file:
            combined_contents = json.load(json_file)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    yes_count_dict = {
        "Video Caption": 0,
        "Action Identification": 0,
        "Object Identification": 0,
        "Spatial Relationship": 0,
        "Action Ordering": 0,
        "Action Temporal Localization": 0,
        "Action Segment Summarization": 0,
        "Action Segmentation and Summarization": 0,
        "Task Success Detection": 0,
        "Task Planning": 0,
    }
    no_count_dict = {
        "Video Caption": 0,
        "Action Identification": 0,
        "Object Identification": 0,
        "Spatial Relationship": 0,
        "Action Ordering": 0,
        "Action Temporal Localization": 0,
        "Action Segment Summarization": 0,
        "Action Segmentation and Summarization": 0,
        "Task Success Detection": 0,
        "Task Planning": 0,
    }
    score_sum_dict = {
        "Video Caption": 0,
        "Action Identification": 0,
        "Object Identification": 0,
        "Spatial Relationship": 0,
        "Action Ordering": 0,
        "Action Temporal Localization": 0,
        "Action Segment Summarization": 0,
        "Action Segmentation and Summarization": 0,
        "Task Success Detection": 0,
        "Task Planning": 0,
    }

    for key, result in combined_contents.items():
        # Computing score
        if not isinstance(result, list):
            continue
        count += 1
        if not result[0]:
            continue
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score
        result[1]["question_type"] = str(result[1]["question_type"]).replace('[', '').replace(']', '').replace("'", "")
        score_sum_dict[result[1]["question_type"]] += score

        # Computing accuracy
        pred = result[0]['pred'] if score < 3 else "yes"
        if "yes" in pred.lower():
            yes_count_dict[result[1]["question_type"]] += 1
            yes_count += 1
        elif "no" in pred.lower():
            no_count_dict[result[1]["question_type"]] += 1
            no_count += 1

    for key, value in yes_count_dict.items():
        type_count = yes_count_dict[key] + no_count_dict[key]
        if type_count == 0:
            continue
        average_score = score_sum_dict[key] / (type_count)
        accuracy = yes_count_dict[key] / type_count
        print(f"{key} Accuracy:", accuracy * 100, " %")
        print(f"{key} Average score:", average_score)
    try:
        average_score = score_sum / count
    except Exception as e:
        print(e)
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy * 100, " %")
    print("Average score:", average_score)


if __name__ == "__main__":
    main()

