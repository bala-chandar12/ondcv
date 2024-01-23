import os
from PIL import Image
import pathlib
import textwrap
import cv2
import numpy as np

import google.generativeai as genai
#import cv2
from matplotlib import pyplot as plt

# Used to securely store your API key
#from google.colab import userdata

from IPython.display import display
from IPython.display import Markdown
import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
HOME = os.getcwd()
print(HOME)

CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(CONFIG_PATH, WEIGHTS_PATH,'cpu')
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
import os
import supervision as sv
from torchvision.ops import box_convert
import cv2 as cv2
def predic(imh):

    IMAGE_NAME =imh
    IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)
    TEXT_PROMPT = "items"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device='cpu'
    )
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    print(detections)
    image = image_source.copy()
    colors = np.random.randint(0, 255, size=(len(detections), 3), dtype=np.uint8)
    nob = len(detections.xyxy)
    for i, detection in enumerate(detections.xyxy):

        image = image_source.copy()
        x_min, y_min, x_max, y_max = detection
        color = (255, 255, 255)  # White

        # Convert float coordinates to integer
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Draw rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, -1)
        #####
        # Get the center of the bounding box
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

        # Calculate the position to center the text
        text = f"{i + 1}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_position = (center_x - text_size[0] // 2, center_y + text_size[1] // 2)

        # Draw text on the image
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        imbox = f"C:/Users/nagar/PycharmProjects/ondc/content/imbox_{i + 1}.jpg"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(imbox, image)
        #####
        # text=f"{i+1}"
        # text_position = (x_min + 18, y_min + 50)
        # cv2.putText(image,text,text_position,cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Display the image with bounding boxes in Colab
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)  # Wait for any key to be pressed
    # cv2.destroyAllWindows()

    # Save the image with bounding boxes
    # imageboxes = "C:/Users/nagar/PycharmProjects/ondc/content/image_with_boxes.jpg"
    # cv2.imwrite(imageboxes, image)


    # Assuming 'image_bgr' is your image and 'detections.xyxy' is your detections array
    # Load your image - replace 'image_path.jpg' with your image path or use your image directly
    # image_bgr = img

    img = Image.open(imh)
    image_bgr = np.array(img)

    # Convert detections to integers for slicing
    detections = [[int(coord) for coord in detection] for detection in detections.xyxy]

    # Crop and save each detected object
    rt=[]
    for i, detection in enumerate(detections):
        x_min, y_min, x_max, y_max = detection
        cropped_object = image_bgr[y_min:y_max, x_min:x_max]  # Crop the object from the image

        # Convert BGR to RGB
        cropped_object_rgb = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)

        # Save each cropped object
        filename = f"C:/Users/nagar/PycharmProjects/ondc/output/object_{i + 1}.jpg"  # Change the filename as needed
        cv2.imwrite(filename, cropped_object_rgb)  # Save the cropped object

        # Display each cropped object using matplotlib
        # plt.figure(figsize=(5, 5))
        # plt.imshow(cropped_object_rgb)
        # plt.axis('off')
        # plt.title(f"Object {i+1}")
        # plt.show()
    # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
    GOOGLE_API_KEY = "AIzaSyCkeJUUMZGkD9DxXoz_XfsK7JF8a3NPfMQ"

    genai.configure(api_key=GOOGLE_API_KEY)
    for i in range(nob):
        img = Image.open(f'C:/Users/nagar/PycharmProjects/ondc/content/imbox_{i + 1}.jpg')
        print("image shape is")
        cv2.imread("")


        #mode = genai.GenerativeModel('gemini-pro-vision')

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Object {i + 1}")
        plt.show()
        from openai import OpenAI

       
        os.environ["OPENAI_API_KEY"] = "api key"

        client = OpenAI()
        encoded_image = encode_image(f"C:/Users/nagar/PycharmProjects/ondc/content/imbox_{i + 1}.jpg")

        # OpenAI.OPENAI_API_KEY = os.environ['']

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"what is the product with white box and number {i+1} inside it and what is it's price?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        t=response.choices[0]

        print(f"Object_{i+1}:{response.choices[0]}")
        #response = mode.generate_content(
         #   [f"What is the name of product and price for the white box background with blue coloured numeral {i + 1}?",
          #   img], stream=True)
        #response.resolve()
        #print(f"Object_{i + 1}:{response.text}")
        temp=[]
        temp.append(t)
        temp.append(f"C:/Users/nagar/PycharmProjects/ondc/output/object_{i + 1}.jpg")
        rt.append(temp)
    return rt

k=predic("C:/Users/nagar/PycharmProjects/ondc/content/electrical.jpg")
print(k)



#from google.colab.patches import cv2_imshow  # Required to display images in Colab

# Load the image (Replace 'image_source' with your image variable)
 # Replace image_source with your image array

# Define detections as a list of bounding box coordinates [x_min, y_min, width, height]
# Assuming detections.xyxy returns the bounding box coordinates
# Replace this with your actual detection coordinates

# Generate random colors for each bounding box


# Draw bounding boxes on the image
#print(to_markdown(response.text))
