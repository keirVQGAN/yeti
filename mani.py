import base64, json, random, subprocess, os, requests, openai, hashlib
import shutil
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from IPython.display import display, Image
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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

def create(manifestos_text, keywordA, keywordB, keywordC, keywordD, model='gpt-3.5-turbo'):
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
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()

def title(manifesto_text, model='gpt-3.5-turbo'):
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
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message.content.strip().replace('"', '')

def image(MANI_TITLE, animal, api_key, prompt, init_image, width=512, height=512, samples=2, negative_prompt='Repeated Edge, Tiling', mask_image=None, prompt_strength=None, num_inference_steps=30, guidance_scale=3, enhance_prompt='yes', seed=None, webhook=None, track_id=None):
    url = 'https://stablediffusionapi.com/api/v3/img2img'
    headers = {'Content-Type': 'application/json'}
    output_dir = f'/content/drive/MyDrive/mani/out/manifestos/{MANI_TITLE}'
    data = {
        "key": api_key,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_image": init_image,
        "width": str(width),
        "height": str(height),
        "samples": str(samples),
        "num_inference_steps": str(num_inference_steps),
        "guidance_scale": guidance_scale,
        "enhance_prompt": enhance_prompt,
        "seed": seed,
        "webhook": webhook,
        "track_id": track_id
    }
    
    if mask_image:
        data["mask_image"] = mask_image
    
    if prompt_strength:
        data["prompt_strength"] = prompt_strength
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        # save response data to a unique JSON file
        filename = f'{animal}.json'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(response_data, f)
        
        # download the generated images
        image_urls = response_data['output']
        image_paths = []
        for url in image_urls:
            response = requests.get(url)
            filename = f'{animal}.jpg'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            image_paths.append(filepath)
            
        return image_paths
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

def display_info(MANI_TITLE, MANI_TEXT, VALUE, LANDSCAPE, ELEMENT, ANIMAL, MANI_SAMPLE, MANI_SAMPLE_LINKS, image_paths):
    console = Console()
    table_width = int(console.width * 0.75)

    main_table = Table(width=table_width, pad_edge=False, collapse_padding=True)
    main_table.add_column(MANI_TITLE, style="white on black")

    # Main text
    main_table.add_row(MANI_TEXT)

    # Table for keywords
    keywords_table = Table(show_header=False, collapse_padding=True)
    keywords_table.add_column("Keyword", style="bold white on black")
    keywords_table.add_column("Value", style="white on black")
    keywords_table.add_row("Value", VALUE)
    keywords_table.add_row("Landscape", LANDSCAPE)
    keywords_table.add_row("Element", ELEMENT)
    keywords_table.add_row("Animal", ANIMAL)
    main_table.add_row(keywords_table)

    # Sample text
    main_table.add_row("Sample Text:")
    main_table.add_row(MANI_SAMPLE)

    # Sample links
    main_table.add_row("Sample Links:")
    links_str = "\n".join(f"[{link}]({link})" for link in MANI_SAMPLE_LINKS)
    main_table.add_row(Markdown(links_str))

    # Display first image in the list, resized to fit the table width
    first_image_path = image_paths[0]
    display(Image(filename=first_image_path, width='400', height='400', metadata={"padding": 10, "border": "white"}))

    console.print(main_table)

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

def html(title, text, image_path):
    """
    Create an HTML file for a manifesto with the specified title, text, and image.

    Args:
    - title (str): The title of the manifesto.
    - text (str): The main text of the manifesto.
    - image_path (str): The path to the image file.

    Returns:
    - None
    """
    # Load the HTML template
    with open('/content/drive/MyDrive/mani/in/html/manifesto_template.html') as f:
        html_template = f.read()

    # Wrap each paragraph in <p> tags
    paragraphs = [f'<p>{p.strip()}</p>' for p in text.split('\n\n')]

    # Load the image
    with PILImage.open(image_path) as img:
        # Resize the image to fit inside the box and maintain the aspect ratio
        width, height = img.size
        aspect_ratio = height / width
        new_width = 300
        new_height = int(new_width * aspect_ratio)
        img = img.resize((new_width, new_height))

        # Convert the image to a base64-encoded string for embedding in the HTML
        with open('temp_image.png', 'wb') as f:
            img.save(f, format='png')
        with open('temp_image.png', 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Remove the temporary image file
        os.remove('temp_image.png')

    # Replace the placeholders with the specified title, text, and image
    html_content = html_template.replace("{{ title }}", title).replace("{{ text }}", ''.join(paragraphs)).replace("{{ image }}", f'data:image/png;base64,{image_data}')

    # Save the HTML content to a file with a unique name based on the title
    filename = f'{title.lower().replace(" ", "_")}.html'
    with open(f'/content/drive/MyDrive/mani/out/manifestos/{title}/{filename}', 'w') as f:
        f.write(html_content)

def sync(folder_path):
    master_folder = "/content/drive/MyDrive/mani/out/master"
    file_extensions = {"html", "json", "pdf", "jpg"}

    # Create the subfolders if they don't exist
    for ext in file_extensions:
        os.makedirs(os.path.join(master_folder, ext), exist_ok=True)

    # Walk through the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = file.split(".")[-1]

            if file_ext in file_extensions:
                destination_folder = os.path.join(master_folder, file_ext)
                destination_file_path = os.path.join(destination_folder, file)

                # Only copy the file if it doesn't exist in the destination folder
                if not os.path.exists(destination_file_path):
                    shutil.copy(file_path, destination_file_path)
                    print(f"Copied {file_path} to {destination_file_path}")
                else:
                    print(f"File {file} already exists in {destination_file_path}. Skipping.")

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_hash = hashlib.md5(file_data).hexdigest()
    return file_hash

def sync_folder(folder_path):
    master_folder = "/content/drive/MyDrive/mani/out/master"
    file_extensions = {"html", "json", "pdf", "jpg"}

    # Create the subfolders if they don't exist
    os.makedirs(os.path.join(master_folder, "manifestos"), exist_ok=True)

    for ext in file_extensions:
        os.makedirs(os.path.join(master_folder, "manifestos", ext), exist_ok=True)

    # Walk through the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = file.split(".")[-1]

            if file_ext in file_extensions:
                destination_folder = os.path.join(master_folder, "manifestos", file_ext)
                destination_file_path = os.path.join(destination_folder, file)

                # Only copy the file if it doesn't exist in the destination folder
                if not os.path.exists(destination_file_path):
                    shutil.copy(file_path, destination_file_path)
                    # print(f"Copied {file_path} to {destination_file_path}")
                else:
                    # Check if file size or hash is different
                    source_file_size = os.path.getsize(file_path)
                    destination_file_size = os.path.getsize(destination_file_path)
                    source_file_hash = get_file_hash(file_path)
                    destination_file_hash = get_file_hash(destination_file_path)

                    if source_file_size != destination_file_size or source_file_hash != destination_file_hash:
                        # Add a unique number as a suffix to the file name
                        unique_number = 1
                        while True:
                            new_file_name = f"{os.path.splitext(file)[0]}_{unique_number:02d}.{file_ext}"
                            new_destination_file_path = os.path.join(destination_folder, new_file_name)

                            if not os.path.exists(new_destination_file_path):
                                shutil.copy(file_path, new_destination_file_path)
                                # print(f"Copied {file_path} to {new_destination_file_path} with unique number")
                                break

                            unique_number += 1
                    else:
                        print(f"File {file} already exists in {destination_file_path}. Skipping.")

def create_pdf(title, text, image_path):

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
        name="SubtitleStyle", fontName="Inter-Medium", fontSize=16, alignment=1, spaceAfter=0.1, leading=18
    )
    body_style = ParagraphStyle(
        name="BodyStyle", fontName="Inter-Light", fontSize=12, alignment=4, spaceAfter=1, leading=14
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
