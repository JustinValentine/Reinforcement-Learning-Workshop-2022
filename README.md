# Reinforcement-Learning-Workshop-2022

# Installation & Setup
### Setting up the virtual envorment 
- Lets create a new virtual enviorment to house our new project called **OpenAiGym** by typing the following comand into the terminal `conda create -n OpenAiGym python=3.9`
- Next we will active our enviorment `conda activate OpenAiGym`
- If you do not have miniconda installed you can get it [here](https://docs.conda.io/en/latest/miniconda.html) 

### Setting up the virtual enviorment kernel for Jupyter Notebook
- First we need to install the following package `pip install --user ipykernel`
- Next we need to add the kernel so we can have it in our Jupyter Notebook `python -m ipykernel install --user --name=OpenAiGym`
- Later if you wanna remove the enverment later use `jupyter kernelspec uninstall myenv`

### Installation
- Firstly we need to install the base gym library `pip install gym` 
- Next we will need to install the atari enviorment dependences `pip install gym[atari]`
- You can freely download Atari 2600 roms [here](#http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) but the Breakout ROM that we will be using is provided 
- Next we will use ALE to import our ROM `ale-import-roms roms/`


### More info
- A good artical to help you get started with OpenAi Gym is [here](https://blog.paperspace.com/getting-started-with-openai-gym/)
