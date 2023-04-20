import base64, json, random, subprocess
from io import BytesIO
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime
import openai

def save_json(title, text, keywords, example_text, date_time, links):
    """Save the manifesto information to a JSON file and to an individual JSON file for each manifesto."""
    # Prepare manifesto data
    manifesto_data = {
        "title": title,
        "text": text,
        "keywords": keywords,
        "example_text": example_text,
        "date_time": date_time,
        "link": links
    }
    
    # Create the JSON output file path for the single manifesto
    single_json_output_path = f"/content/drive/MyDrive/mani/out/manifestos/{title}/{title}.json"

    # Save the single manifesto data to a JSON file
    with open(single_json_output_path, 'w') as f:
        json.dump(manifesto_data, f, indent=4)

    # Create the JSON output file path for the main JSON file
    main_json_output_path = '/content/drive/MyDrive/mani/out/json/manifesto_data.json'

    # Load existing data or create an empty list if the main JSON file does not exist
    existing_data = []
    if os.path.exists(main_json_output_path):
        with open(main_json_output_path, 'r') as f:
            existing_data = json.load(f)

    # Check if the new manifesto data is a duplicate
    is_duplicate = any(data["title"] == title for data in existing_data)

    # Append the new manifesto data to the existing data if it's not a duplicate
    if not is_duplicate:
        existing_data.append(manifesto_data)

        # Write the updated data back to the main JSON output file
        with open(main_json_output_path, 'w') as f:
            json.dump(existing_data, f, indent=4)


def sample(manifestos_limit=5, sample_limit=500):
    """Select a random sample of manifestos and return their links and sampled text."""
    # Load the manifestos data
    with open('/content/drive/MyDrive/mani/in/json/manifestos.json') as f:
        manifestos_data = json.load(f)

    # Select a random sample of manifestos with a minimum text length of 50 characters
    selected_manifestos = random.sample(
        [m for m in manifestos_data if len(m.get("main_text", "").strip()) >= 50], manifestos_limit
    )

    # Create a list of links to the selected manifestos
    links = [m["link"] for m in selected_manifestos]

    # Combine the sample of the main text of the selected manifestos
    manifestos_text = "\n\n".join([
        "Manifesto sample for '" + m["title"] + "': " + get_manifesto_sample(m.get("main_text", ""), sample_limit)
        for m in selected_manifestos if m.get("main_text", "").strip()
    ])

    return links, manifestos_text

def get_manifesto_sample(text, sample_limit):
    """Return a sample of the text with a maximum length of sample_limit."""
    words = text.split()
    sample = ''
    i = 0

    # Add words to the sample until the sample_limit is reached
    while i < len(words) and len(sample + words[i]) < sample_limit:
        sample += words[i] + ' '
        i += 1

    # If the entire text is within the sample_limit
    if i == len(words):
        return sample.rstrip()
    else:
        # If the last word has a period
        if '.' in words[i]:
            while i < len(words) and len(sample + words[i]) + 1 < sample_limit:
                sample += words[i] + ' '
                i += 1
            return (sample.rstrip() + '...').rstrip()
        else:
            # Go back to the last word with a period
            while i > 0 and '.' not in sample:
                i -= 1
                sample = ' '.join(words[:i]) + ' '
            return (sample.rstrip() + '...').rstrip()

def create(manifestos_text, keywordA, keywordB, keywordC, keywordD):
    messages = [
        {
            "role": "system",
            "content": "You are a highly creative and skilled writer. Craft an engaging and captivating manifesto by considering different perspectives, emotions, and vivid imagery."
        },
        {
            "role": "user",
            "content": f"Create a two-paragraph manifesto based on this example: {manifestos_text}. It should focus on care, sustainability, and the significance of indigenous and more-than-human knowledge, using these <mark>keywords</mark>: {keywordA}, {keywordB}, {keywordC}. The manifesto should be poetic, accessible, and aligned with contemporary writing styles, and should not exceed 250 words. Start with a 7-word subtitle, followed by the main text. The manifesto will help the recipient reconnect with the more-than-human world through their more-than-human guide ({keywordD})."
        }
    ]
    response = openai.ChatCompletion.create(model='gpt-4', messages=messages)
    return response.choices[0].message.content.strip()

def title(manifesto_text):
    messages = [
        {
            "role": "system",
            "content": "You are an expert at creating concise and powerful titles for written works."
        },
        {
            "role": "user",
            "content": f"Generate a title for the following manifesto using no more than three words and no non-alphabet characters. Consider the essence and emotions conveyed by the manifesto: {manifesto_text[:50]}..."
        }
    ]
    response = openai.ChatCompletion.create(model='gpt-4', messages=messages)
    return response.choices[0].message.content.strip().replace('"', '')

def select_image(image_dir, color_space):
    """
    Select an image from a directory based on the specified color space.

    Args:
    - image_dir (str): The directory containing the images.
    - color_space (str): The desired color space of the selected image.

    Returns:
    - The path of the selected image file.
    """
    # Get list of all image files in directory
    images = os.listdir(image_dir)

    # Select an image based on the color space
    if color_space == 'RGB':
        image_path = os.path.join(image_dir, [f for f in images if f.endswith('.png')][0])
    elif color_space == 'CMYK':
        image_path = os.path.join(image_dir, [f for f in images if f.endswith('.jpg')][0])

    return image_path

def html(title, text):
    """
    Create an HTML file for a manifesto with the specified title, text, and image.

    Args:
    - title (str): The title of the manifesto.
    - text (str): The main text of the manifesto.
    - image_dir (str): The directory containing the images.

    Returns:
    - None
    """
    # Load the HTML template
    with open('/content/drive/MyDrive/mani/in/html/manifesto_template.html') as f:
        html_template = f.read()
  
    # Remove the first sentence from the text
    text = text.split('. ', 1)[1]

    # Wrap each paragraph in <p> tags
    paragraphs = [f'<p>{p.strip()}</p>' for p in text.split('\n\n')]

    # Get a random image from the specified directory
    image_dir = '/content/drive/MyDrive/mani/in/images/RGB'
    images = [file for file in os.listdir(image_dir) if file.endswith('.png')]
    if images:
        image_path = os.path.join(image_dir, random.choice(images))
        with Image.open(image_path) as img:
            # Get the aspect ratio of the image
            width, height = img.size
            aspect_ratio = height / width

            # Resize the image to fit inside the box and maintain the aspect ratio
            new_width = 300
            new_height = int(new_width * aspect_ratio)
            img = img.resize((new_width, new_height), resample=Image.LANCZOS)

            # Convert the image to a base64-encoded string for embedding in the HTML
            with open('temp_image.png', 'wb') as f:
                img.save(f, format='png')
            with open('temp_image.png', 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Remove the temporary image file
            os.remove('temp_image.png')
    else:
        image_data = ''

    # Replace the placeholders with the specified title, text, and image
    html_content = html_template.replace("{{ title }}", title).replace("{{ text }}", ''.join(paragraphs)).replace("{{ image }}", f'data:image/png;base64,{image_data}')

    # Save the HTML content to a file with a unique name based on the title
    filename = f'{title.lower().replace(" ", "_")}.html'
    with open(f'/content/drive/MyDrive/mani/out/manifestos/{title}/{filename}', 'w') as f:
        f.write(html_content)   

import random
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def create_pdf(title, text):
    # Randomly select an image from the specified folder
    image_folder = "/content/drive/MyDrive/mani/in/images/CMYK"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
    image_path = os.path.join(image_folder, random.choice(image_files))

    # Prepare output file path
    output_folder = f"/content/drive/MyDrive/mani/out/manifestos/{title}"
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"{output_folder}/{title}.pdf"

    # Create the document with reduced top margin
    doc = SimpleDocTemplate(output_file, pagesize=A4, topMargin=inch * 0.3)

    # Register custom fonts
    font_folder = "/content/drive/MyDrive/mani/fonts"
    pdfmetrics.registerFont(TTFont("Oswald-Bold", f"{font_folder}/oswald/Oswald-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Regular", f"{font_folder}/inter/Inter-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Light", f"{font_folder}/inter/Inter-Light.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Medium", f"{font_folder}/inter/Inter-Medium.ttf"))

    # Prepare styles
    title_style = ParagraphStyle(
        name="TitleStyle", fontName="Oswald-Bold", fontSize=36, alignment=1, spaceAfter=0.5, textTransform='uppercase', leading=38
    )
    subtitle_style = ParagraphStyle(
        name="SubtitleStyle", fontName="Inter-Medium", fontSize=16, alignment=1, spaceAfter=1, leading=21
    )
    body_style = ParagraphStyle(
        name="BodyStyle", fontName="Inter-Light", fontSize=10, alignment=4, spaceAfter=1, leading=12
    )

    # Add image
    image = RLImage(image_path, 250 * 4 / 3, 250 * 4 / 3)
    image.Align = 'CENTER'

    # Add title
    title_paragraph = Paragraph(title, title_style)

    # Split text into subtitle and body
    subtitle, body = text.split('\n', 1)

    # Add subtitle
    subtitle_paragraph = Paragraph(subtitle, subtitle_style)

    # Add body text
    body = body.replace('\n', '<br/>')  # Replace newlines with HTML line breaks
    body_paragraph = Paragraph(body, body_style)

    # Build document
    flowables = [image, Spacer(1, 12), title_paragraph, Spacer(1, 12), subtitle_paragraph, Spacer(1, 12), body_paragraph]
    doc.build(flowables)
            

