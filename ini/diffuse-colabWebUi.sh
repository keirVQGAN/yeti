user_token = "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"
user_header = f"\"Authorization: Bearer {user_token}\""
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
wget --header={user_header} https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O /content/stable-diffusion-webui/model.ckpt
COMMANDLINE_ARGS="--exit" REQS_FILE="requirements.txt" python launch.py
cd stable-diffusion-webui
git pull
COMMANDLINE_ARGS="--share --gradio-debug --gradio-auth yeti:yeti" REQS_FILE="requirements.txt" python launch.py
