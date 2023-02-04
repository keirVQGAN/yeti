# Yeti // Functions 'Aug 22
import csv
import cv2
import glob
import imageio
import os
import os.path
import shutil
import time
from IPython.display import clear_output
from IPython.display import Image, display
from PIL import Image
from dirsync import sync
from rich.console import Console
from pathlib import Path
import pandas as pd
import json
from base64 import b64decode
from fpdf import FPDF
import openai
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

console = Console ( )

# --------------------------------------------------------------------FUNCTIONS
# Ken
# -----------------------------------------------------------------------------
def ken(inPath, outPath, WIDTH, FPS, SHIFT,ZOOM, SECONDS, ZOOMmain):
    # -------------------------------------------------------------------------
    kenPathIn = f'{inPath}/ken'
    kenPathOut = f'{outPath}/ken'
    kenInImages=glob.glob(f'{kenPathIn}/*')
    kenOutVideo=[]
    for image in kenInImages:
      imageBase=Path(image).stem
      mk(f'{kenPathOut}/{imageBase}/{WIDTH}/{FPS}fps')
      imageName=f'{kenPathOut}/{imageBase}/{WIDTH}/{FPS}fps/{imageBase}_s{SHIFT}_z{ZOOM}-zm{ZOOMmain}-{SECONDS}s.mp4'
      kenOutVideo.append(imageName)
    return kenOutVideo, kenInImages

  
# -----------------------------------------------------------------------------
def kenConf(WIDTH, SHIFT, ZOOM, FPS, LENGTH, ZOOMmain):
  # ---------------------------------------------------------------------------
  LINES=[70,71,85,86,91,97]
  REPLACES=[f'	intWidth = min(int({WIDTH} * fltRatio), {WIDTH})',
  f'	intHeight = min(int({WIDTH} / fltRatio), {WIDTH})',
  f"		'fltShift': {SHIFT},",
  f"		'fltZoom': {ZOOM},",
  f"		'fltSteps': numpy.linspace(0.0, {ZOOMmain}, {LENGTH}).tolist(),",
  f"	moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult [1:] ], fps={FPS}).write_videofile(arguments_strOut)"]

  for LINE,REPLACE in zip(LINES,REPLACES):
    FILE='/content/3d-ken-burns/autozoom.py'
    overWrite(FILE,LINE,REPLACE)
    
    
# -----------------------------------------------------------------------------
def overWrite(FILE,LINE,REPLACE):
    # -------------------------------------------------------------------------
    REPLACEN=f'{REPLACE}\n'
    with open(FILE, 'r', encoding='utf-8') as file:
        data = file.readlines()

    data[LINE-1] = REPLACEN

    with open(FILE, 'w', encoding='utf-8') as file:
        file.writelines(data)

        
# --------------------------------------------------------------------FUNCTIONS
# CONSOLE
# -----------------------------------------------------------------------------
def txtH(heading) :
    # -------------------------------------------------------------------------
    console.print ( f"<< [bright_white] {heading} [/bright_white] >>" )


# -----------------------------------------------------------------------------
def txtB(action) :
    # -------------------------------------------------------------------------
    console.print ( f"[r black]>> {action}[/r black]" )


# -----------------------------------------------------------------------------
def txtW(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">> [bright_white]{action}[/bright_white]" )


# -----------------------------------------------------------------------------
def txtC(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">> [bright_cyan]{action}[/bright_cyan] | [r black]{details}[/r black]" )


# -----------------------------------------------------------------------------
def txtM(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">> [bright_magenta]{action}[/bright_magenta] | [r black]{details}[/r black]" )


# -----------------------------------------------------------------------------
def txtY(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">> [bright_yellow]{action}[/bright_yellow] | [r black]{details}[/r black]" )


# -----------------------------------------------------------------------------
def conSettings(localPath , drivePath , gpu) :
    # ---------------------------------------------------------------------------- 
    txtC ( 'Local Path' , localPath )
    txtC ( 'Drive Path' , drivePath )
    txtY ( '>> CUDA GPU ' , gpu [ 1 ] )

    
# -----------------------------------------------------------------------------
def listPath(path):
    # -----------------------------------------------------------------------------
    pathList=glob.glob(f'{path}/*')
    return pathList
    


# ----------------------------------------------------------------------------
def slug(s) :
    valid_chars = "-_. %s%s" % (string.ascii_letters, string.digits)
    file = ''.join(c for c in s if c in valid_chars)
    file = file.replace(' ','_')
    return file


# -----------------------------------------------------------------------------    
def csv2ls(csv_file) :
    # -------------------------------------------------------------------------
    with open ( csv_file , 'r' , encoding = 'utf-8-sig' ) as f :
        reader = csv.reader ( f )
        list1 = [ rows [ 0 ] for rows in reader ]

    return list1 [ 1 : ]


# -----------------------------------------------------------------------------
def mk(path) :
    # -------------------------------------------------------------------------
    if not os.path.exists ( path ) :
        os.makedirs ( path )

        
# -----------------------------------------------------------------------------
def lsName(path):
    # -------------------------------------------------------------------------
    l=os.listdir(path)
    li=[x.split('.')[0] for x in l]
    return li


# -----------------------------------------------------------------------------
def imagePath(path , ext , scale) :
    # -------------------------------------------------------------------------
    from IPython.display import Image , display
    for file in os.listdir ( path ) :
        if file.endswith ( "*.ext" ) :
            txtH ( file )
            display ( Image ( filename = os.path.join ( path , file ) , width = scale ) )


# -------------------------------------------------------------------------------
def montage(inpath , outpath , label=True) :
    # ---------------------------------------------------------------------------
    file_paths = [ ]
    for root , directories , files in os.walk ( inpath ) :
        for filename in files :
            filepath = os.path.join ( root , filename )
            file_paths.append ( filepath )
            sorted ( file_paths )
    montPaths = " ".join ( file_paths )
    if label :
        montSettings = f"""-label '%f' -font Helvetica -pointsize 12 -background '#000000' -fill 'gray' -define jpeg:size=175x175 -geometry 175x175+2+2 -auto-orient {montPaths} {outpath}"""
    if not label :
        montSettings = f"""-background '#000000' -fill 'gray' -define jpeg:size=175x175 -geometry 175x175+2+2 -auto-orient {montPaths} {outpath}"""
    return montSettings , montPaths


# -----------------------------------------------------------------------------
def timeTaken(start_time) :
    # -----------------------------------------------------------------------------
    import time
    timeTakenFloat = "%s seconds" % (time.time ( ) - start_time)
    timeTaken = timeTakenFloat
    timeTaken_str = str ( timeTaken )
    timeTaken_split = timeTaken_str.split ( '.' )
    timeTakenShort = timeTaken_split [ 0 ] + '' + timeTaken_split [ 1 ] [ :0 ]
    txtM ( 'Complete:' , f'{timeTakenShort} Seconds' )


# -------------------------------------------------------------------
def copyExt(src , dest , ext) :
    # -----------------------------------------------------------------
    for file_path in glob.glob ( os.path.join ( src , '**' , ext ) , recursive = True ) :
        new_path = os.path.join ( dest , os.path.basename ( file_path ) )
        shutil.copy ( file_path , new_path )


# -------------------------------------------------------------------
def moveExt(src , dest , ext) :
    # -----------------------------------------------------------------
    for file_path in glob.glob ( os.path.join ( src , '**' , ext ) , recursive = True ) :
        new_path = os.path.join ( dest , os.path.basename ( file_path ) )
        shutil.move ( file_path , new_path )


# -------------------------------------------------------------------
def fps(video_file) :
    # -------------------------------------------------------------------
    cap = cv2.VideoCapture ( video_file )
    frame_count = int ( cap.get ( cv2.CAP_PROP_FRAME_COUNT ) )
    return frame_count

# --------------------------------------------------------------------------
def outSync(localPath,outPath,timeNow):
    # -----------------------------------------------------------------------
    localImages=f'{localPath}/images_out'
    localMasks=f'{localPath}/masks'
    localConfig=f'{localPath}/config'
    drive=f'{outPath}/txt2img/{timeNow}'
    driveMasks=f'{drive}/masks'
    driveImages=f'{drive}/images'
    driveConfig=f'{drive}/config'
    
    LOCAL=[localImages,localMasks,localConfig]
    DRIVE=[driveImages,driveMasks,driveConfig]
    
    for d,l in zip(DRIVE,LOCAL):
        mk(d)
        sync(l,d,'sync')

# --------------------------------------------------------------------------
def thresh(imagePath , outPath, imageNames) :
   # -----------------------------------------------------------------------
    threshMasked = [ ]
    imageName = os.path.splitext(imageNames)[0]
    for thresh in range ( 20 , 221 , 10 ) :
        img = cv2.imread ( imagePath )
        ret , img_binary = cv2.threshold ( img , thresh , 255 , cv2.THRESH_BINARY )
        OUT=f'{outPath}/masks/{imageName}'
        mk(OUT)
        outMask=f'{OUT}/{imageName}-{thresh}_mask.jpg'
        imageio.imwrite ( outMask , img_binary )
        threshMasked.append ( outMask )
    txtC('Threshold Masks Made for', imagePath)
    return threshMasked, imageName

def yml(QUALITY, csv_file, confPath, init_image):
    # --------------------------------------------------------------------------
    # QUALITY SETTINGS
    if QUALITY == 'test':
      _width = 300
      _cut_outs = 128
      _cut_pow = 2.8
      _pixel_size = 2
      _direct_init_weight = 0.75
      _gradient_accumulation_steps = 2
      _steps_per_scene = 500
      _save_every = 100
      _display_every = 10
      _clear_every = 20
      _display_scale = 1

    if QUALITY == 'draft':
      _width = 300
      _cut_outs = 320
      _cut_pow = 2.8
      _pixel_size = 2
      _direct_init_weight = 0.75
      _gradient_accumulation_steps = 1
      _steps_per_scene = 500
      _save_every = 100
      _display_every = 20
      _clear_every = 100
      _display_scale = 0.75

    if QUALITY == 'proof':
      _width = 350
      _cut_outs = 100
      _cut_pow = 2.4
      _pixel_size = 2
      _direct_init_weight = 0.75
      _gradient_accumulation_steps = 2
      _steps_per_scene = 500
      _save_every = 100
      _display_every = 10
      _clear_every = 20
      _display_scale = 1

    # --------------------------------------------------------------------------
    # MAKE yaml from csv
#     shutil.rmtree(confPath)
    mk(confPath)
    df = pd.read_csv(csv_file)
    col_names = list(df.columns.values)
    for col in col_names:
        globals()[col] = []
        for value in df[col]:
            globals()[col].append(value)            
    for names, preffixs, scenes, suffixs, styles in zip(name, preffix, scene, suffix, style):
        yaml = f'{confPath}/{names}.yaml'
        yaml_settings = f"""# @package _global_\nfile_namespace: {names}-{QUALITY}\nscene_prefix: {preffixs} \nscenes: {scenes}\nwidth: {_width}\ncutouts: {_cut_outs}\ncut_pow: {_cut_pow}\npixel_size: {_pixel_size}\ndirect_init_weight: {_direct_init_weight}\ngradient_accumulation_steps: {_gradient_accumulation_steps}\nsteps_per_scene: {_steps_per_scene}\nsave_every: {_save_every}\ndisplay_every: {_display_every}\nclear_every: {_clear_every}\nscene_suffix: {suffixs}\ndisplay_scale: {_display_scale}\ninit_image: {init_image}"""

        f = open(yaml, 'w')
        f.write(yaml_settings)
        

def grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def prompter(jsonFile,_prompt):
    with open(jsonFile) as jsonPrompt:
      data = json.load(jsonPrompt)
      from promptgen import PromptGenerator
      prompt = PromptGenerator(_prompt, data)
      text_prompt, strength, prompt_data = prompt.generate()
      clear_output()
      return data, text_prompt, strength, prompt_data
# --------------------------------------------------------------------------
############################################################################
#openAI
# -----------------------------------------------------------------------------
def TXT(txtPath='/content/texts',N=1,ENGINE='text-ada-001',PROMPT='A poem about snails eating cats',TEMPREATURE=1,MAX_TOKENS=64,TOP_P=0.88,FREQUENCY_PENALTY=0, PRESENCE_PENALTY=0,STOP='""'):
    # -----------------------------------------------------------------------------
    response = openai.Completion.create(
    engine=ENGINE,
    prompt=PROMPT,
    temperature=TEMPREATURE,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    frequency_penalty=FREQUENCY_PENALTY,
    presence_penalty=PRESENCE_PENALTY,
    stop=STOP,
    n=N
    )
    textOut=response.choices[0].text
    textClean1 = textOut.lstrip('\n')
    textClean2 = textClean1.lstrip('\t')
    textClean = textClean2.rstrip('\n')
    # txtTXT(textOut)

    file_name_txt = f'{txtPath}/{PROMPT[:10]}-{response["created"]}.txt'
    with open(file_name_txt, 'w') as f:
      f.write(textClean)


    return file_name_txt

# -----------------------------------------------------------------------------
def IMG(PROMPT="A snail covered in iridescent feathers",VARIATIONS=1,SIZE="256x256",FORMAT="b64_json"):
  # ---------------------------------------------------------------------------
    imageOut = openai.Image.create(
      prompt=PROMPT,
      n=VARIATIONS,
      size=SIZE,
      response_format=FORMAT
  )
  #Save as a JSON file
    file_name = imagePath / f"{PROMPT[:10]}-{imageOut['created']}.json"
    with open(file_name, mode="w", encoding="utf-8") as file:
      json.dump(imageOut, file)
  #Encode JSON file as an image
    for index, image_dict in enumerate(imageOut["data"]):
      image_data = b64decode(image_dict["b64_json"])
      image_file = imagePath / f"{file_name.stem}-{index}.png"
      with open(image_file, mode="wb") as png:
          png.write(image_data)
          display(Image(image_file))
          return image_file

# -----------------------------------------------------------------------------    
def PDF(TEXT, IMAGE, TEXT_WIDTH=100, TEXT_HEIGHT=100, PDF_OUT='untitled.pdf'):
  # ---------------------------------------------------------------------------
    pdf = FPDF()
    pdf.add_page('P,A5')
    with open(TEXT, 'rb') as fh:
      txt = fh.read().decode('latin-1')
    # a=IMAGE
    pdf.image(str(IMAGE),w=TEXT_WIDTH,h=TEXT_HEIGHT)
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(180, 6, txt, 0,'M')
    pdf.output(PDF_OUT, 'F')



# END OF SCRIPT##############################################################
############################################################################
####################################################################yeti2022
