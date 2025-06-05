import os
import numpy as np
import cv2
import tensorflow_datasets as tfds
import json

from tqdm import tqdm

def save_video(images, output_path, fps=30, convert_to_brg=True):
    if isinstance(images, np.ndarray):
        if len(images.shape) == 4:  # Check for (frames, height, width, channels)
            height, width = images.shape[1:3]
        else:
            raise ValueError("Numpy array should have 4 dimensions (frames, height, width, channels).")
    elif isinstance(images, list):
        if len(images) > 0 and isinstance(images[0], np.ndarray):
            height, width = images[0].shape[:2]
        else:
            raise ValueError("List should contain numpy arrays with shape (height, width, channels).")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in images:
        if convert_to_brg:
            image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = frame
        out.write(image_bgr)
    out.release()

def get_camera_image(step, dataset_name):
    image = step["observation"]["images"].numpy()
    return image


def get_question_answering(step):
    question = step["observation"]["raw_text_question"].numpy().decode('utf-8')
    answer = step["observation"]["raw_text_answer"].numpy().decode('utf-8')
    return question, answer

def check_valid_string(str):
    if not re.search(r'^[a-zA-Z]+( [a-zA-Z]+)*$', str) and not re.search(r'^[a-zA-Z]+( [a-zA-Z]+)*\.?$', str):
        return False
    else:
        return True

def filter_condition(instruction, dataset_name):
    # Check if the action is a valid string (contains at least one valid word)
    if dataset_name in ["columbia_cairlab_pusht_real", "utokyo_xarm_pick_and_place_converted_externally_to_rlds"]:
        return True
    else:
        if not check_valid_string(instruction):
            return False
        else:
            return True


def format_question_with_video(question=None):
    video_token = '<image>'
    formatted_question = f"{video_token}\n{question}" if question is not None else f"{video_token}\n"
    return formatted_question


def get_instance_template():
    instance = {
        "video": "video_name.png",
        "conversations": [
            {
                "from": "human",
                "value": "question"
            },
            {
                "from": "gpt",
                "value": "answer"
            }
        ]}
    return instance


def main():
    dataset_name = 'robot_vqa'
    print(f"Dataset [{dataset_name}] Processing .......")
    base_dir = '/data/robot_vqa'
    video_dir = f"/data/Pretrain/{dataset_name}"
    os.makedirs(video_dir, exist_ok=True)
    video_index = 0
    total_episode, missing_episode = 0, 0

    qa_annotation = []
    episodes = tfds.builder_from_directory(base_dir).as_dataset(split='all')
    for episode in tqdm(episodes):
        total_episode += 1
        images, instructions = [], []
        is_filter = True
        for step in episode["steps"]:
            images = get_camera_image(step=step, dataset_name=dataset_name)
            question, answer = get_question_answering(step=step)
            instance = get_instance_template()
            instance['video'] = f"{video_index:06}.mp4"
            instance['conversations'][0]['value'] = format_question_with_video(question=question)
            instance['conversations'][1]['value'] = answer
            qa_annotation.append(instance)

        save_video(images=images, output_path=os.path.join(video_dir, f"{video_index:06}.mp4"))
        video_index += 1
        if video_index == 200000:
            break

    new_instance_number = len(qa_annotation) // 1000
    new_name = f'{dataset_name}_{new_instance_number}K.json'
    annotation_dir = f"/data/Pretrain/{new_name}"
    with open(annotation_dir, 'w') as file:
        json.dump(qa_annotation, file, indent=4)


if __name__ == '__main__':
    main()