from transformers import BlipProcessor,BlipForConditionalGeneration
from PIL import Image

#Initialize the processor and model from hugging face
processor = BlipProcessor("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the image
image = Image.open("path_to_the_image.py")

# Prepare the image
inputs = processor(image,return_tensors="pt")

# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)

print("Generated Caption:",caption)