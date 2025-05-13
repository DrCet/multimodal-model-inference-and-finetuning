from dataclasses import dataclass, field
from typing import Union, Dict
import torch
from transformers import (
    HfArgumentParser,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor
)
from qwen_omni_utils import process_mm_info
import sys 
import os
import logging
import re
import soundfile as sf 
from IPython.display import Audio, display, clear_output
import time
import ipywidgets as widgets



logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='Qwen/Qwen2.5-Omni-3B',
        metadata={'help':'Hugging face model name or local model path'}
    )
    cache_dir: str = field(
        default=f'./{model_name_or_path}-cache',
        metadata={'help':'Where to cache model and data'}
    )
    device_map: str = field(
        default='auto',
        metadata={'help':'Devices to load model'}
    )
    torch_dtype: str = field(
        default='auto',  # Fixed: string type, valid dtype
        metadata={'help': 'Torch dtype (float16, bfloat16, float32, auto)'}
    )
    weights_only: bool = field(
        default=False,
        metadata={'help':'Default weights_only=False'}
    )

@dataclass
class ProcessingArguments:
    add_generation_prompt: bool = field(
        default=True,
        metadata={'help':'Default add_generation_prompt=True'}
    )
    tokenize: bool = field(
        default=False,
        metadata={'help':'Default tokenize=False'}
    )
    use_audio_in_video: bool = field(
        default=False,
        metadata={'help':'Default use_audio_in_video=False'}
    )
    speaker:str = field(
        default='Chelsie',
        metadata={'help':'Default speaker="Chelsie"'}
    )
    return_audio: bool = field(
        default=True,
        metadata={'help':'Default return_audio=True'}
    )


# Function to check Flash Attention 2 compatibility
def is_flash_attention_2_supported():
    try:
        # Check CUDA availability and compute capability
        if not torch.cuda.is_available():
            print("CUDA not available.")
            return False
        compute_capability = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        if compute_capability < 80:  # Need compute capability >= 8.0
            print(f"GPU compute capability {compute_capability/10} is not supported (requires >= 8.0).")
            return False

        # Check PyTorch and CUDA versions
        torch_version = torch.__version__.split("+")[0]
        torch_major, torch_minor = map(int, torch_version.split(".")[:2])
        cuda_version = torch.version.cuda
        cuda_major, cuda_minor = map(int, cuda_version.split(".")[:2]) if cuda_version else (0, 0)
        if torch_major < 2 or (torch_major == 2 and torch_minor < 2):
            print(f"PyTorch {torch_version} is not supported (requires >= 2.2).")
            return False
        if cuda_major < 11 or (cuda_major == 11 and cuda_minor < 7):
            print(f"CUDA {cuda_version} is not supported (requires >= 11.7).")
            return False

        # Check Flash Attention availability
        if not torch.backends.cuda.flash_sdp_enabled():
            print("FlashAttention-2 is not available. Ensure flash-attn >= 2.1.0 is installed.")
            return False

        print("FlashAttention-2 is supported.")
        return True
    except Exception as e:
        print(f"Error checking FlashAttention-2 compatibility: {e}")
        return False
    


def main():

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

    # 1. Parser
    parser = HfArgumentParser((ModelArguments, ProcessingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('json'):
        model_args, processing_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, processing_args = parser.parse_args_into_dataclasses()


    # 2. Load model and processor
    attn_implementation = "flash_attention_2" if is_flash_attention_2_supported() else "sdpa"
    logger.info(f"Using attention implementation: {attn_implementation}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        device_map=model_args.device_map,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=attn_implementation,
        weights_only=model_args.weights_only
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_args.model_name_or_path,

    )

    def prepare_inputs(conversation=None, elements: Dict = None):
        prompt_template = {
            'role':'user',
            'content':[
                {'type':'text', 'text':elements['text']}
            ]
        }

        # Add images to prompt
        for image in elements['images']:
            if is_image_file(image):
                prompt_template['content'].append({'type':'image', 'image':image})

        # Add audio to prompt
        for audio in elements['audio']:
            if is_audio_file(image):
                prompt_template['content'].append({'type':'audio', 'audio':audio})

        # Add video to prompt
        for video in elements['video']:
            if is_video_file(image):
                prompt_template['content'].append({'type':'video', 'image':video})



        text = processor.apply_chat_template(conversation, add_generation_prompt=processing_args.add_generation_prompt, tokenize=processing_args.tokenize)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=processing_args.use_audio_in_video)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=processing_args.use_audio_in_video)
        # Move inputs to model dtype and device
        inputs = inputs.to(model.device).to(model.dtype)
        return conversation, inputs

    def is_image_file(path):
        """Check if path is an image file."""
        return path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))

    def is_audio_file(path):
        """Check if path is an audio file."""
        return path.lower().endswith(('.wav', '.mp3', '.ogg', '.aac', '.flac'))

    def is_video_file(path):
        """Check if path is a video file."""
        return path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))

    def extract_prompt_elements(prompt, verbose=False):
        """Extract text, image, audio, and video from prompt."""
        # Regex patterns
        image_pattern = re.compile(r'((https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.(jpg|jpeg|png|gif|bmp))($|[^\w]))')
        audio_pattern = re.compile(r'((https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.(wav|mp3|ogg|aac|flac))($|[^\w]))')
        video_pattern = re.compile(r'((https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.(mp4|avi|mov|wmv|mkv))($|[^\w]))')

        elements = {
            "images": [],
            "audio": [],
            "video": [],
            "text": prompt
        }

        # Extract URLs and files
        for pattern, key in [
            (video_pattern, "video"),
            (audio_pattern, "audio"),
            (image_pattern, "images"),
        ]:
            matches = pattern.findall(prompt)
            for match in matches:
                url_or_path = match[0]  # Full match (without boundary)
                elements[key].append(url_or_path)
                # Remove from prompt to isolate text
                elements["text"] = re.sub(pattern, ' ', elements["text"])

        # Clean up text (remove extra spaces)
        elements["text"] = ' '.join(elements["text"].split())
        if verbose:
            print("Extracted elements:", elements)

        return elements
    
    def chat():
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            }
        ]

        # Create chat interface
        prompt_widget = widgets.Text(
            value='',
            placeholder='Type your prompt (e.g., Describe image1.jpg and image2.jpg)',
            description='You:',
            layout={'width': '500px'}
        )
        submit_button = widgets.Button(
            description='Submit',
            button_style='primary',
            tooltip='Click to submit prompt'
        )
        output = widgets.Output()

        def on_submit(button):
            with output:
                clear_output()
                prompt = prompt_widget.value.strip()
                if prompt.lower() == 'exit':
                    print("Chat ended.")
                    return

                elements = extract_prompt_elements(prompt, verbose=True)
                conversation_copy = conversation.copy()
                conv, inputs = prepare_inputs(conversation_copy, elements)

                try:
                    logger.info(f"Generating response on device={model.device}")
                    text_ids, audio = model.generate(**inputs, use_audio_in_video=processing_args.use_audio_in_video)
                except Exception as e:
                    print(f"Inference failed: {e}")
                    return

                text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response = text[0].split('assistant\n')[-1]

                os.makedirs('.generated_audio', exist_ok=True)
                output_audio = ".generated_audio/response.wav"
                if audio is not None:
                    sf.write(
                        output_audio,
                        audio.reshape(-1).detach().cpu().numpy(),
                        samplerate=24000
                    )

                print("Assistant:", response)
                if audio is not None:
                    audio_data, sample_rate = sf.read(output_audio)
                    audio_duration = len(audio_data) / sample_rate
                    display(Audio(output_audio, autoplay=True))
                    time.sleep(audio_duration + 0.5)
                prompt_widget.value = ''  # Clear input after submission

        submit_button.on_click(on_submit)
        display(widgets.VBox([prompt_widget, submit_button, output]))

        # Fallback non-interactive mode
        print("If the widget doesn't display, use this cell to test a prompt:")
        test_prompt = "Describe /kaggle/input/your-dataset/image1.jpg and /kaggle/input/your-dataset/image2.jpg"
        print(f"Testing with prompt: {test_prompt}")
        elements = extract_prompt_elements(test_prompt, verbose=True)
        conversation_copy = conversation.copy()
        conv, inputs = prepare_inputs(conversation_copy, elements, processing_args, processor, model)
        try:
            text_ids, audio = model.generate(**inputs, use_audio_in_video=processing_args.use_audio_in_video)
            text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = text[0].split('assistant\n')[-1]
            os.makedirs('.generated_audio', exist_ok=True)
            output_audio = ".generated_audio/test_response.wav"
            if audio is not None:
                sf.write(output_audio, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
            print("Assistant (test):", response)
            if audio is not None:
                display(Audio(output_audio, autoplay=True))
        except Exception as e:
            print(f"Test inference failed: {e}")
    chat()

if __name__ == '__main__':
    main()