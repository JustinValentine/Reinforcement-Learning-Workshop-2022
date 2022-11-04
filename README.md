# Reinforcement-Learning-Workshop-2022
The accompanying slides are [here](https://docs.google.com/presentation/d/1mvGdp7hg0sJhwTD7Dr4b7YCOmsuGGPXnZuCEL-GLZZY/edit#slide=id.p)

In this demo will be using [OpenAi Gym](https://www.gymlibrary.dev/), a standard API for reinforcement learning with a lot of built in environments

# Installation & Setup
### Setting up the virtual envorment 
#### Using Conda 
- Lets create a new virtual enviorment to house our new project called **OpenAiGym** by typing the following comand into the terminal `conda create -n OpenAiGym python=3.9`
- Next we will active our enviorment `conda activate OpenAiGym`
- If you do not have miniconda installed you can get it [here](https://docs.conda.io/en/latest/miniconda.html) 

#### Using venv
- Lets create a new virtual enviorment to house our new project called **OpenAiGym** by typing the following comand into the terminal `python3 -m venv OpenAiGym-env`
- To activate on **Windows** run: `OpenAiGym-env\Scripts\activate.bat`
- To activate on **Unix or MacOS** run: `source OpenAiGym-env/bin/activate`

### Setting up the virtual enviorment kernel for Jupyter Notebook
- First we need to install the following package `pip install --user ipykernel`
- Next we need to add the kernel so we can have it in our Jupyter Notebook `python -m ipykernel install --user --name=OpenAiGym`
- Later if you wanna remove the enverment use `jupyter kernelspec uninstall myenv`

### Installation
- Firstly lets install Jupyter Notebook `pip install notebook`
- Next we need to install the base gym library `pip install gym` 
- We will  also need to install the atari enviorment dependences `pip install 'gym[atari]'`
- You can freely download Atari 2600 roms [here](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) but the Breakout ROM that we will be using is provided 
- Next we will use ALE to import our ROM `ale-import-roms ROMS/`
- Next install imageio for capturing our image frames `pip install imageio`
#### Installing pytorch
- Next install pytorch `conda install pytorch -c pytorch`
- and then all then all the requirements `pip install -r requirements.txt` 

### More info
- A good artical to help you get started with OpenAi Gym is [here](https://blog.paperspace.com/getting-started-with-openai-gym/)
- Another article that was very helpful for setting up the Atari environment is [here](https://blog.devgenius.io/teaching-a-neural-network-to-play-the-breakout-game-793ad7d1b20e)
