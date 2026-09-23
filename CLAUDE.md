# Notes for Claude

Whenever a new zip or new code is delivered to the user, always end the reply with the
Mac terminal commands to update and run it:

```bash
cd ~/Downloads && unzip -o smartroute_code.zip
cd ~/Downloads/smartroute
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "gradio>=4,<5" "huggingface_hub<1.0" folium requests "urllib3<2"
cd src
python app.py
```
Then open http://127.0.0.1:7860.
