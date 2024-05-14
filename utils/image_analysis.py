import json

from PIL import Image, ImageDraw, ImageFont
from IPython import display

def display_image(img_path, action):
    image = Image.open(img_path)
    image = image.convert("RGB")
    if action == 'return':
        return image
    else:
        image.save(r'assets/image.png')
        return

def display_json(json_path, action):
    with open(json_path) as f:
        json_data = json.load(f)

    if action == 'return':
        return json_data
    else:
        for annotation in json_data['form']:
            print(annotation)
        return

def display_annotation_on_image(img, json_data):
    draw = ImageDraw.Draw(img, "RGBA")

    font = ImageFont.load_default()

    label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

    for annotation in json_data['form']:
        label = annotation['label']
        general_box = annotation['box']
        draw.rectangle(general_box, outline=label2color[label], width=2)
        draw.text((general_box[0] + 10, general_box[1] - 10), label, fill=label2color[label], font=font)
        words = annotation['words']
        for word in words:
            box = word['box']
            draw.rectangle(box, outline=label2color[label], width=1)
    
    img.save(r'assets/annotations_on_image.png')
