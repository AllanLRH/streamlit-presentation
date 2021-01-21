# Streamlit presentation

Streamlit presentation for the Datascience chapter

Standing in the project root, launch the file using `streamlit run snippets/00-hello-world.py`.

When developing with Streamlit, check out the [cheatsheet](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py)!

The self-driving car demo can be run by issuing
```
streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/streamlit_app.py
```

## Environment

Create a virtual environment and install everything from _requirements.txt_, and everyting should run fine (tested under Python 3.8):

Windåze/Powershell
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Bash/ZSH
```
python -m venv .venv
source .venv\bin\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
