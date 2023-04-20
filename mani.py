def get_selected_manifestos_text(manifestos_limit=5, sample_limit=500):
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

def create_manifesto(manifesto_text, keywords):
    """
    Generate a manifesto based on provided text and keywords using OpenAI's GPT-3 API.

    Args:
    - manifesto_text (str): The main text of the manifesto.
    - keywords (list of str): A list of keywords to be incorporated into the manifesto.

    Returns:
    - A string containing the generated manifesto.
    """
    # Construct prompt for API request
    prompt = f"Generate a manifesto based on the following text and keywords:\n\n{manifesto_text}\n\nKeywords: {', '.join(keywords)}\n\nManifesto:"
    
    # Send request to OpenAI's GPT-3 API
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated manifesto from the API response
    generated_manifesto = response.choices[0].text.strip()

    return generated_manifesto

def generate_manifesto_title(manifesto_text):
    """
    Generate a title for a manifesto based on the provided text using OpenAI's GPT-3 API.

    Args:
    - manifesto_text (str): The main text of the manifesto.

    Returns:
    - A string containing the generated title.
    """
    # Construct prompt for API request
    prompt = f"Generate a title for a manifesto based on the following text:\n\n{manifesto_text}\n\nTitle:"

    # Send request to OpenAI's GPT-3 API
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated title from the API response
    generated_title = response.choices[0].text.strip()

    return generated_title

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

def create_manifesto_html(title, text, image_dir):
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

def create_pdf(title, text):
    # Define required paths and folders
    image_folder = "/content/drive/MyDrive/mani/in/images/CMYK"
    output_folder = f"/content/drive/MyDrive/mani/out/manifestos/{title}"
    font_folder = "/content/drive/MyDrive/mani/fonts"

    # Randomly select an image from the specified folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
    image_path = os.path.join(image_folder, random.choice(image_files))

    # Prepare output file path
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"{output_folder}/{title}.pdf"

    # Create the document with reduced top margin
    doc = SimpleDocTemplate(output_file, pagesize=A4, topMargin=inch * 0.5)

    # Register custom fonts
    fonts = [("Oswald-Bold", "oswald/Oswald-Bold.ttf"), ("Inter-Regular", "inter/Inter-Regular.ttf"),
             ("Inter-Light", "inter/Inter-Light.ttf"), ("Inter-Medium", "inter/Inter-Medium.ttf")]

    for font_name, font_path in fonts:
        pdfmetrics.registerFont(TTFont(font_name, f"{font_folder}/{font_path}"))

    # Prepare styles
    title_style = ParagraphStyle(
        name="TitleStyle", fontName="Oswald-Bold", fontSize=42, alignment=1, spaceAfter=8, textTransform='uppercase', leading=48
    )
    subtitle_style = ParagraphStyle(
        name="SubtitleStyle", fontName="Inter-Medium", fontSize=14, alignment=1, spaceAfter=1
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

def save_manifesto_to_json(title, text, keywords, example_text, date_time, output_folder, links):
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

