import google.generativeai as genai
from PIL import Image
import io

# Configure your Gemini API key
genai.configure(api_key="AIzaSyB71LHO-PURZoeRp_4mKhMl37HX0ITtNVw")

# Load the image
image_path = "Images\Lilies.jpg"
image = Image.open(image_path)

# Convert to JPEG and get binary data
image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format='JPEG')
image_data = image_byte_arr.getvalue()

# Load the Gemini 1.5 model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Make a request with text a + image
response = model.generate_content([
    "What's in this image?",
    {"mime_type": "image/jpeg", "data": image_data}
])

# Print result
print(response.text)