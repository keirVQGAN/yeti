#Yeti | Oct 22
#Installs pytti5 and required libraries
cd /content                    
pip install -r /content/yeti/requirements_txt2img.txt                        
apt -qq install imagemagic       
git clone --recurse-submodules -j8 https://github.com/pytti-tools/pytti-core 
pip install ./pytti-core/vendor/AdaBins                                      
pip install ./pytti-core/vendor/CLIP                                         
pip install ./pytti-core/vendor/GMA                                          
pip install ./pytti-core/vendor/taming-transformers                          
pip install ./pytti-core                                                     
python -m pytti.warmup                                                       
touch /content/config/conf/empty.yaml
