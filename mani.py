#@markdown > **yeti**Manifesto `mani.py`
import base64, datetime, hashlib, heapq, json, os, random, re, requests, shutil, subprocess
import numpy as np
import openai

from PIL import Image
from io import BytesIO
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from IPython.display import display, Image as IPythonImage
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_hash = hashlib.md5(file_data).hexdigest()
    return file_hash

def manifesto(role, question, keywordA, keywordB, keywordC, keywordD, model, manifestos_limit=5, sample_limit=500):
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def select_top_k_sentences(text, k=5):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(sentences)
        top_k_indices = heapq.nlargest(k, range(len(sentences)), tfidf_matrix.sum(axis=1).take)
        return ' '.join([sentences[i] for i in top_k_indices])

    def sample(manifestos_limit=5, sample_limit=500):
        with open('/content/drive/MyDrive/mani/in/json/manifestos.json') as f:
            manifestos_data = json.load(f)

        manifestos_data = [m for m in manifestos_data if len(m.get("main_text", "").strip()) >= 50]
        preprocessed_texts = [preprocess_text(m.get("main_text", "")) for m in manifestos_data]

        num_clusters = 5
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(preprocessed_texts)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X)

        selected_manifestos = []
        for cluster in range(num_clusters):
            cluster_manifestos = [m for idx, m in enumerate(manifestos_data) if kmeans.labels_[idx] == cluster]
            if cluster_manifestos:
                selected_manifestos.append(random.choice(cluster_manifestos))

        while len(selected_manifestos) < manifestos_limit:
            selected_manifestos.append(random.choice(manifestos_data))

        links = [m["link"] for m in selected_manifestos]

        manifestos_text = "\n\n".join([
            f"Manifesto sample for '{m['title']}': {select_top_k_sentences(m.get('main_text', ''), sample_limit)}"
            for m in selected_manifestos
        ])

        return links, manifestos_text

    manifestos_links, manifestos_text = sample(manifestos_limit, sample_limit)

    messages = [
        {
            "role": "system",
            "content": f'{role}'
        },
        {
            "role": "user",
            "content": f'{question.format(keywordA=keywordA, keywordB=keywordB, keywordC=keywordC, manifestos_text=manifestos_text)}'
        }
    ]

    response = openai.ChatCompletion.create(model=model, messages=messages)
    result = response.choices[0].message.content.strip()

    title, text = result.split('\n', 1)

    data = {
        'role': role,
        'question': question,
        'manifestos_links': manifestos_links,
        'samples': manifestos_limit,
        'characters_sampled': sample_limit,
        'manifestos_text': manifestos_text,
        'keywordA': keywordA,
        'keywordB': keywordB,
        'keywordC': keywordC,
        'keywordD': keywordD,
        'title': title,
        'text': text.strip(),
        'model': model,
        'timestamp': str(datetime.datetime.now())
    }
    out_path='/content/drive/MyDrive/mani/out/manifestos'
    manifesto_dir=f'{out_path}/{title}'
    os.makedirs(manifesto_dir, exist_ok=True)
    return result, data, manifesto_dir
    
def image_prompt(animal, model='gpt-4'):

    roles = [
        f"As an AI image creator, your task is to generate a stunning and realistic photographic portrait of a pagan {animal}. Your objective is to provide a brief and concise description of the image, focusing on the visual and technical elements that should be incorporated into the image to create a captivating and harmonious composition that produces a pagan, neo-rave portrait of the {animal}. Ensure the prompt is informed by the qualities and size of the {animal}"
    ]

    role = random.choice(roles)

    questions = [
        f"Can you provide a one or two sentence prompt for an AI to generate a visually striking and experimental close-up photographic portrait of a pagan {animal} using custom neon and fluorescent lighting, contemporary and innovative techniques, and a neo-rave or pagan aesthetic? The prompt should start with 'A medium format photograph of' and include technical details such as camera type, lighting, perspective, composition, and color palette. The {animal} should be draped in living garlands of tropical flowers, vines, orchids, moss, lichen, fantastical mushrooms, and other plants to capture its unique features and symbolism and shot at a scale that is approrpiate for the {animal}. Please use concise technical terms and include the statement 'shot with a Hasselblad H6c-100c medium format camera.'"
    ]

    question = random.choice(questions)
    
    messages = [
        {
            "role": "system",
            "content": role
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()

def image(title, animal, api_key, prompt, width=512, height=512, samples=2, mask_image=None, prompt_strength=None, num_inference_steps=30, guidance_scale=7, enhance_prompt='no', seed=None, webhook=None, track_id=None):
    def clean_filename(text):
        result = text.replace(' ', '_')
        return result
    negative='/content/drive/MyDrive/mani/in/txt/negative_prompts.txt'
    with open(negative, 'r') as file:
        negative_prompt = file.read()

    headers = {'Content-Type': 'application/json'}
    output_dir = f'/content/drive/MyDrive/mani/out/manifestos/{title}'
    url = 'https://stablediffusionapi.com/api/v3/text2img'
    data = {
        "key": api_key,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
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
        filename = f'{animal}_image.json'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(response_data, f)

        # download the generated images
        image_urls = response_data['output']
        image_paths = []
        for url in image_urls:
            response = requests.get(url)
            filename = f'{animal}.jpg'
            filename = clean_filename(filename)
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            image_paths.append(filepath)

        return image_paths
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

def tjson(data,title):
    file_path_master = '/content/drive/MyDrive/mani/out/json/manifesto_master.json'
    mode = 'a' if os.path.exists(file_path_master) else 'w'
    with open(file_path_master, mode) as f:
        if mode == 'a':
            f.write(',\n')
        json.dump(data, f, indent=4)

    file_path = f'/content/drive/MyDrive/mani/out/manifestos/{title}/{title}_data.json'    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def pdf(title, text, image_path):
    # Prepare output file path
    output_folder = f"/content/drive/MyDrive/mani/out/manifestos/{title}"
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"{output_folder}/{title}.pdf"

    # Create the document with reduced top margin
    doc = SimpleDocTemplate(output_file, pagesize=A4, topMargin=inch * 0.4)

    # Register custom fonts
    font_folder = "/content/drive/MyDrive/mani/fonts"
    pdfmetrics.registerFont(TTFont("Oswald-Bold", f"{font_folder}/oswald/Oswald-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Regular", f"{font_folder}/inter/Inter-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Light", f"{font_folder}/inter/Inter-Light.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Medium", f"{font_folder}/inter/Inter-Medium.ttf"))

    # Prepare styles
    title_style = ParagraphStyle(
        name="TitleStyle", fontName="Oswald-Bold", fontSize=28, alignment=1, spaceBefore=5, spaceAfter=1.5, textTransform='uppercase', leading=38
    )
    body_style = ParagraphStyle(
        name="BodyStyle", fontName="Inter-Light", fontSize=11, alignment=4, spaceAfter=1, leading=14
    )

    # Add image
    image = RLImage(image_path, 230 * 4 / 3, 230 * 4 / 3)
    image.Align = 'CENTER'

    # Add title
    title_paragraph = Paragraph(title, title_style)

    # Add body text
    body = text.replace('\n', '<br/>')  # Replace newlines with HTML line breaks
    body_paragraph = Paragraph(body, body_style)

    # Build document
    flowables = [image, Spacer(1, 12), title_paragraph, Spacer(1, 12), body_paragraph]
    doc.build(flowables)

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

    # Convert the text to HTML
    body_html = ''.join([f'<p>{p.strip()}</p>' for p in text.split('\n\n')])

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
    html_content = html_template.replace("{{ title }}", title).replace("{{ text }}", body_html).replace("{{ image }}", f'data:image/png;base64,{image_data}')

    # Save the HTML content to a file with a unique name based on the title
    filename = f'{title.lower().replace(" ", "_")}.html'
    with open(f'/content/drive/MyDrive/mani/out/manifestos/{title}/{filename}', 'w') as f:
        f.write(html_content)
        
def sync(folder_path):
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

def process_manifesto(role, question, keywordA, keywordB, keywordC, keywordD, model, sample, characters, animal, stable_api):

    # Call the create_manifesto function
    result, data, manifesto_dir = manifesto(role, question, keywordA, keywordB, keywordC, keywordD, model, sample, characters)

    # Set the variables for creating html, pdf and json files.
    result = data
    role = result['role']
    question = result['question']
    manifestos_links = result['manifestos_links']
    manifestos_text = result['manifestos_text']
    keywordA = result['keywordA']
    keywordB = result['keywordB']
    keywordC = result['keywordC']
    keywordD = result['keywordD']
    title = result['title']
    text = result['text']
    model = result['model']
    timestamp = result['timestamp']
    print(title)

    # Generate prompt and image based on manifesto and animal
    prompt = image_prompt(animal, model)
    image_paths = image(title, animal, stable_api, prompt)
    image_path = image_paths[0]
    data['image_path'] = image_path
    print(image_path)

    tjson(data, title)
    pdf(title, text, image_path)
    html(title, text, image_path)
    sync(manifesto_dir)

    pdf_path = f"/content/drive/MyDrive/mani/out/manifestos/{title}/{title}.pdf"
    html_path = f"/content/drive/MyDrive/mani/out/manifestos/{title}/{title}.html"
    json_path = f"/content/drive/MyDrive/mani/out/manifestos/{title}/{title}_data.json"

    return title, text, image_path, pdf_path

