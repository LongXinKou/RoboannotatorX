import torch
from io import BytesIO
import base64
import torch
import math
import ast

from PIL import Image
from decord import VideoReader, cpu
from transformers import StoppingCriteria

from constants import IMAGE_TOKEN_INDEX

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0

        # Tokenize each keyword and store the token ids
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        # Record the length of input before generation starts
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]

        # Try to match any keyword exactly in the recent token sequence
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        # If no exact match, decode the last `offset` tokens into string
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

def process_video_with_decord(video_path, image_processor, video_fps=1, video_stride=2):
    """
    Process a video file using Decord to extract and preprocess frames.

    Args:
        video_path (str): Path to the video file.
        image_processor: A processor object (e.g., from a vision-language model) to preprocess video frames.
        video_fps (int, optional): Target number of frames per second to sample from the video.
                                   If <= 0, use 'stride' instead to sample every N-th frame.
        stride (int, optional): Used when video_fps is 0, defines the interval to sample frames.

    Returns:
        video (torch.Tensor): Preprocessed video frames with shape [T, C, H, W], where T is the number of frames.
        total_frame_num (int): The total number of frames in the original video.
    """
    # Load the video using Decord
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)  # Get total number of frames in the video

    # Raise an error if the video contains only a single frame
    if total_frame_num == 1:
        raise ValueError("Single frame video detected")

    # Sample frame indices based on target fps
    if video_fps > 0:
        # Calculate the sampling interval
        fps = round(vr.get_avg_fps() / video_fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
    else:
        # Sample every 'stride'-th frame
        frame_idx = list(range(0, total_frame_num, video_stride))

    # Extract the frames at the selected indices and convert to numpy array
    video = vr.get_batch(frame_idx).asnumpy()

    # Preprocess the extracted frames (e.g., resize, normalize)
    video = image_processor.preprocess(video, return_tensors='pt')['pixel_values']

    return video, total_frame_num

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(images, image_processor, image_aspect_ratio=None, image_grid_pinpoints="[(224, 224), (336, 336),]"):
    """
    Process a list of input images based on the specified aspect ratio handling method.

    Args:
        images (List[str or PIL.Image.Image]): A list of image file paths or PIL.Image objects.
        image_processor: An image processor object (e.g., from a vision transformer pipeline).
        image_aspect_ratio (str, optional): Defines how to handle image aspect ratio.
            Options are:
                - 'pad': Pad the image to a square shape.
                - 'anyres': Resize/pad based on a grid of resolutions and divide into patches.
                - None or other: Use the processor's default behavior.
        image_grid_pinpoints (str or list): List of resolutions (e.g., [(224, 224), (336, 336)])
            used for 'anyres' processing mode. Can be a string or a list of (width, height) tuples.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]:
            - If all images have the same shape, returns a single stacked tensor (B, C, H, W).
            - Otherwise, returns a list of tensors.
    """
    processed_images = []
    for image in images:
        if isinstance(image, str):  # If it's a file path
            image = Image.open(image).convert('RGB')
        processed_images.append(image)

    new_images = []

    # Step 2: Handle different aspect ratio processing strategies
    if image_aspect_ratio == 'pad':
        for image in processed_images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)

    elif image_aspect_ratio == "anyres":
        for image in processed_images:
            image = process_anyres_image(image, image_processor, image_grid_pinpoints)
            new_images.append(image)

    else:
        return image_processor(processed_images, return_tensors='pt')['pixel_values']

    # Step 3: Stack if all shapes match
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)

    return new_images