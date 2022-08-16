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
from dirsync import sync
from rich.console import Console

console = Console ( )

# -----------------------------------------------------------------------------
def kenConf(WIDTH, SHIFT, ZOOM, FPS):
  # -----------------------------------------------------------------------------
  LINES=[70,71,85,86,91,97]
  REPLACES=[f'	intWidth = min(int({WIDTH} * fltRatio), {WIDTH})',
  f'	intHeight = min(int({WIDTH} / fltRatio), {WIDTH})',
  f"		'fltShift': {SHIFT},",
  f"		'fltZoom': {ZOOM},",
  f"		'fltSteps': numpy.linspace(0.0, 1.0, {LENGTH}).tolist(),",
  f"	moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps={FPS}).write_videofile(arguments_strOut)"]

  for LINE,REPLACE in zip(LINES,REPLACES):
    FILE='/content/3d-ken-burns/autozoom.py'
    yeti.overWrite(FILE,LINE,REPLACE)
    
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
    console.print ( f"<<[bright_white] {heading} [/bright_white]>>" )


# -----------------------------------------------------------------------------
def txtB(action) :
    # -------------------------------------------------------------------------
    console.print ( f"[r black]>>{action}[/r black]" )


# -----------------------------------------------------------------------------
def txtW(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">>[bright_white]{action}[/bright_white]" )


# -----------------------------------------------------------------------------
def txtC(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">>[bright_cyan]{action}[/bright_cyan] | [r black]{details}[/r black]" )


# -----------------------------------------------------------------------------
def txtM(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">>[bright_magenta]{action}[/bright_magenta] | [r black]{details}[/r black]" )


# -----------------------------------------------------------------------------
def txtY(action , details) :
    # -------------------------------------------------------------------------
    console.print ( f">>[bright_yellow]{action}[/bright_yellow] | [r black]{details}[/r black]" )


# -----------------------------------------------------------------------------
def conSettings(localPath , drivePath , gpu) :
    # ---------------------------------------------------------------------------- 
    txtC ( 'Local Path' , localPath )
    txtC ( 'Drive Path' , drivePath )
    txtY ( '>> CUDA GPU ' , gpu [ 1 ] )

    
# ----------------------------------------------------------------------------
def slug(s) :
    valid_chars = "-_. %s%s" % (string.ascii_letters, string.digits)
    file = ''.join(c for c in s if c in valid_chars)
    file = file.replace(' ','_')
    return file


# -----------------------------------------------------------------------------    
def csv2ls(csv_file) :
    # --------------------------------------------------------------------------
    with open ( csv_file , 'r' , encoding = 'utf-8-sig' ) as f :
        reader = csv.reader ( f )
        list1 = [ rows [ 0 ] for rows in reader ]

    return list1 [ 1 : ]


# -----------------------------------------------------------------------------
def mk(path) :
    # --------------------------------------------------------------------------
    if not os.path.exists ( path ) :
        os.makedirs ( path )


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
# WRITE // Threshold Masks
def thresh(imagePath , outPath) :
    imageName = os.path.splitext ( imagePath ) [ 0 ]
    threshMasked = [ ]
    for thresh in range ( 20 , 221 , 10 ) :
        img = cv2.imread ( image )
        ret , img_binary = cv2.threshold ( img , thresh , 255 , cv2.THRESH_BINARY )
        imageio.imwrite ( f'{outPath}/{imageName}/{thresh}_mask.jpg' , img_binary )
        threshMasked.append ( f'{outPath}/{imageName}/{thresh}_mask.jpg' )

# --------------------------------------------------------------------------
############################################################################
# END OF SCRIPT##############################################################
############################################################################
####################################################################yeti2022
