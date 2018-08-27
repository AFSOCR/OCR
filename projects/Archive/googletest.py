from Image_Preprocessing import get_string
import io
from google.cloud import vision
vision_client = vision.ImageAnnotatorClient()
file_name = '/home/afsocr/notebooks/Images/02_Car_Transmission_Hand.jpg'

with io.open(file_name, 'rb') as image_file:
    content = image_file.read()
    image = vision_client.image(content=content)
    
labels = image.detect_labels()

for label in labels:
    print(label.description)
