from transformers import GenerationConfig, AutoImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from PIL import Image

# Define the directory where your local checkpoint is stored
config_directory = "kaggle/working/checkpoint-10000"

# Load the model from the local checkpoint
model = VisionEncoderDecoderModel.from_pretrained(config_directory)

# Load the generation configuration from the local directory
generation_config = GenerationConfig.from_pretrained(config_directory)
print(generation_config)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
#image_processor = AutoImageProcessor.from_pretrained(config_directory)
# Load the image and process it
image_path = "the.png"
image = Image.open(image_path).convert("RGB")
pixel_values = image_processor(image, return_tensors="pt").pixel_values

# Generate text from the image
# Make sure to set the parameters correctly for generating multiple words/lines
generated_ids = model.generate(pixel_values, max_length=100, num_beams=4, early_stopping=True)

# Decode the generated text
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

"""
from transformers import GenerationConfig, AutoImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import torch

# Define the directory where your local checkpoint is stored
config_directory = "testocr.png"

# Load the model from the local checkpoint
model = VisionEncoderDecoderModel.from_pretrained(config_directory)

# Load the generation configuration from the local directory
generation_config = GenerationConfig.from_pretrained(config_directory)
print(generation_config)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

# Load the image and process it
image_path = "ocr_handwriting_reco_adrian_sample.webp"
image = Image.open(image_path).convert("RGB")
pixel_values = image_processor(image, return_tensors="pt").pixel_values

# Generate text from the image
generated_ids = model.generate(pixel_values)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# Assuming the model provides bounding box coordinates, we simulate it here
# Replace this with actual coordinates from your model if available
bbox_coordinates = [(50, 50, 250, 100)]  # Example bounding box coordinates (x1, y1, x2, y2)

# Draw bounding boxes and text on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for bbox in bbox_coordinates:
    draw.rectangle(bbox, outline="red", width=2)
    draw.text((bbox[0], bbox[1] - 10), generated_text, fill="red", font=font)

# Save or display the image
output_image_path = "annotated_image.png"
image.save(output_image_path)
image.show()
"""