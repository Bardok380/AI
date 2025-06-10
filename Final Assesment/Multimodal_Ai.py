import os
import torch
import whisper
import openai
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Initialize models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization")
open.api_key = os.getenv("OPENAI_API_KEY")

# === Image to text (Captioning) ===
def image_to_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Audio to Text (Speech Recognition)
def audio_to_text(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

# Video to Text Summary (Simple - transcript the summarize)
def video_to_summary(video_audio_path):
    transcript = audio_to_text(video_audio_path)
    summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Text to Image (OpenAI DALL·E)
def text_to_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['date'][0]['url']
    return image_url

# Feedback Incorporation (Logging + Placeholder for Model Update)
def store_user_feedback(task_type, input_data, feedback):
    with open("feedback_log.txt", "a") as f:
        f.write(f"TASK: {task_type}\nInput: {input_data}\nFEEDBACK: {feedback}\n---\n")

# Example Usage
if __name__ == "__main__":
    print("Multimodal AI System Example")

    img_caption = image_to_caption("sample.jpg")
    print("Image Caption:", img_caption)

    audio_text =audio_to_text("sample.wav")
    print("Audio Transcription:", audio_text)

    video_summary = video_to_summary("sample_video_audio.wav")
    print("Video Summary:", video_summary)

    img_url = text_to_image("A futuristic city skyline at sunset")
    print("Generated Image URL:", img_url)

    store_user_feedback("image_captioning", "sample.jpg", "More detail on surroundings")
    