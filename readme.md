<div align="center">

# 🔥🔥🔥Fooocus-ControlNet-SDXL🔥🔥🔥

</div>

| [![Open In Colab](asset/discord-icon-svgrepo-com.svg)](https://discord.gg/evHG9KEcxE) |
|---------------------------------------------------------------------------------------|

---

# 🚀 Updates

* **[2023.10.29]** release  [Mask_Inpaint](#MaskInpaint)
---

![image prompt (1)](https://github.com/fenneishi/Fooocus-ControlNet-SDXL/assets/33345538/566422f4-208e-4b88-a449-54a02f1e9712)

---

![image](asset/ip_depth/ip_depth.png)

---

![image](asset/canny/canny.png)

---

![image](asset/depth/depth.png)

---

![image](asset/ip/ip.png)

---

![image](asset/pose_face/pose_face.png)

---

![image](asset/recolor/recolor.png)

---

![image](asset/sketch/sketch.png)

---

![image](asset/compare.png)

---

Fooocus-ControlNet-SDXL is a ⭐free⭐ image generating software (based on [Fooocus](https://github.com/lllyasviel/Fooocus)
, [ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly)
,👉[SDXL](https://github.com/Stability-AI/generative-models) , [IP-Adapter](https://ip-adapter.github.io/) ,
etc.).

Fooocus-ControlNet-SDXL adds more control to the original Fooocus software.

---

| control                                                                                              | status 🚀                             | show case            |
|------------------------------------------------------------------------------------------------------|---------------------------------------|----------------------|
| image prompt                                                                                         | ✅ provided by fooocus                 | [ip](#Image)         |
| [canny](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-canny)                   | ✅provided by fooocus                  | [canny](#Canny)      |
| cpds                                                                                                 | ✅ provided by fooocus                 |                      |
| [depth](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-depth)                   | ✅ provided by Fooocus-ControlNet-SDXL | [depth](#Depth)      |
| [pose(body,hand,face)](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-openpose) | ✅ provided by Fooocus-ControlNet-SDXL | [pose](#Pose)        |
| recolor                                                                                              | ✅ provided by Fooocus-ControlNet-SDXL | [recolor](#Recolor)  |
| sketch                                                                                               | ✅ provided by Fooocus-ControlNet-SDXL | [sketch](#Sketch)    |
| [segmentation](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-segmentation)     | 📍 todo                               |                      |
| [pose(only body)](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-openpose)      | 📍 todo                               |                      |
| [pose(only hand)](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-openpose)      | 📍 todo                               |                      |
| [pose(only face)](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-openpose)      | 📍 todo                               |                      |
| [pose(body+hand)](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-openpose)      | 📍 todo                               |                      |
| [pose(body+face)](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-openpose)      | 📍 todo                               |                      |
| [Scribble](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-scribble)             | 📍 todo                               |                      |
| [Soft Edges](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-soft-edge)          | 📍 todo                               |                      |
| [Linear](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-lineart)                | 📍 todo                               |                      |
| [Anime Linear](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-anime-lineart)    | 📍 todo                               |                      |
| [Tile](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-tile)                     | 📍 todo                               |                      |
| [relighing]()                                                                                        | 📍 todo                               |                      |
| [mask InPaint]()                                                                                     | ✅ provided by Fooocus-ControlNet-SDXL | [mask](#MaskInpaint) |
| [newBackground]()                                                                                    | 📍 todo                               |                      |
| etc                                                                                                  | 📍 todo                               |                      |

---

### 💡 Fooocus-ControlNet-SDXL keeps it simple

- Fooocus is an excellent SDXL-based software, which provides excellent generation effects based on the simplicity of
  liking midjourney, while being free as stable diffusion.
- FooocusControl inherits the core design concepts of fooocus, in order to minimize the learning threshold,
  FooocusControl has the same UI interface as fooocus (only in the Input Image/Image Prompt/advance to add more
  options).
- FooocusControl does all the complicated stuff behind the scenes, such as model downloading, loading, registration,
  image preprocessing, etc. Users don't need to bother with any of this at all, you just need to check the desired image
  control method.

### 💡 Fooocus-ControlNet-SDXL facilitates secondary development

- Fooocus-ControlNet-SDXL simplifies the way fooocus integrates with controlnet by simply defining pre-processing and adding
  configuration files.
- If you are a developer with your own unique controlnet model , with Fooocus-ControlNet-SDXL , you can easily integrate it into
  fooocus .
- In addition to controlnet, FooocusControl plans to continue to integrate ip-adapter and other models to further
  provide users with more control methods.

### 💡 Fooocus-ControlNet-SDXL pursues the 📍out-of-the-box use of software📍

- Free software usually encounters a lot of installation and use of the problem, such as 😞 network problems caused by the
  model file that can not be downloaded and updated 😞, 😞a variety of headaches gpu driver😞, 😞plug-ins lack of dependent
  libraries and other issues😞. These are very annoying invisible thresholds. fooocusControl is committed to solving these
  problems.
- FooocusControl has no external dependencies, all the dependencies are built into the software.
- FooocusControl provides both online(light) and offline(massive) download methods, the offline(massive) version will download all the models built into
  the software, which is very friendly to users who need to run offline, or those who have a bad network.
- Outside the windows platform, FooocusControl will try to use docker and other technologies to further simplify the
  user installation, to avoid various gpu and dependency problems to the greatest extent possible.
- FooocusControl will add multiple download sources for each model (coming soon) to prevent models from being downloaded
  or updated due to network problems.

### 💡 中国用户友好

- FooocusControl将会针对每一个模型添加多个下载源(即将上线)，防止因网络问题导致模型无法下载或者更新..
- windows平台上offline下载方式基本避免网络问题
- 中文翻译(即将上线)
- 代码码云镜像(https://gitee.com/fenneishi/Fooocus-ControlNet-SDXL)

---

# [🔖Free Install Fooocus-ControlNet-SDXL](#%EF%B8%8Finstall%EF%B8%8F)

---

# <center>🍇show case🍇</center>

# Sketch

Using a sketch image as a prompt input to👉👉👉generate an image incorporating sketch elements.Awesome!!! Can be used for anything now!
![image](asset/sketch/snip.png)

---

# ImagePrompt

Using an image as a prompt input➕one-sentence description👉👉👉A perfect image.
![image](asset/ip/snip.png)

# ImagePrompt+Depth

Background image controls the backgroundposture image➕ controls the pose➕one-sentence description👉👉👉You can achieve any 3D result with the background and pose you desire.
![image](asset/ip_depth/snip.png)

---

# Canny

Picture as a prompt, AI extracts 3D wireframe information from the image➕one-sentence description👉👉👉A picture that perfectly aligns with 3D wireframe information.
![image](asset/canny/snip.png)

---

# Depth

Picture as a prompt, AI extracts 3D information from the image ➕ one-sentence description👉👉👉A perfectly 3D-informed image.
![image](asset/depth/snip.png)

---

# Pose

:face
Using a facial pose from an image as a prompt to control input👉👉👉it generates an image with a specific facial pose consistent with it. This is perfect for making model images and design illustrations, it's simply unbeatable!
![image](asset/pose_face/snip.png)

---

# Recolor

Using an image as a prompt input to👉👉👉re-color it.
![image](asset/recolor/snip.png)

# MaskInpaint

![image](asset/inpaint_outpaint/mask_inpaint.png)

---

# <center>🛠️install🛠️</center>

### Windows

<details>
<summary>Click here to the see</summary>

##### 1️⃣ download the software

| version                                    | Description                                                                                            | Suitable For                                                                 | download(Unzip Password:ddert657)                                                                                                           |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| <span style="color:green;"> online </span> | update automatically, download model automatically <span style="color:green;">when needed</span>       | for users with a <span style="color:green;">good </span> internet connection | [>>> Click here to download from huggingface <<<]() |
|                                            |                                                                                                        |                                                                              | [>>> Click here to download from BaiduNetDisk <<<](https://pan.baidu.com/s/1H65eLWgeYXTos9aTUJqOKg?pwd=t0hj)                                |
|                                            |                                                                                                        |                                                                              | [>>> Click here to download from GoogleDrive <<<](https://drive.google.com/file/d/12Evu9H5MENnCL8xTBciiNwfkyStm9CcR/view?usp=sharing)       |
| <span style="color:red;"> offline </span>  | without update, <span style="color:red;">pre-download all models</span>  with the installation package | for users with a <span style="color:red;">bad </span> internet connection    | >>> Click here to download from huggingface <<< (Uploading to Hugging Face always fails😞)                                                  |
|                                            |                                                                                                        |                                                                              | [>>> Click here to download from BaiduNetDisk <<<](https://pan.baidu.com/s/173m6TWu8KZzijVCIO7VMuQ?pwd=1qsd)                                |
|                                            |                                                                                                        |                                                                              | [>>> Click here to download from GoogleDrive <<<](https://drive.google.com/file/d/1u8OavD2v5DrSaIdz61C8bI-qrq1shi7A/view?usp=sharing)       |

PS : offline is the old version,please use online version

##### 2️⃣ unzip the file(Unzip Password:ddert657)

##### 3️⃣ click 'run.bat' to run the software

![image](asset/run_bat.png)

##### 4️⃣ 👏👏👏 Having fun👏👏👏

##### Q&A

<details>
<summary>Q: Do I need to download the original fooocus software?</summary>
No, you don't need to. Fooocus-ControlNet-SDXL is a standalone software, not a fooocus plugin. 
Like most other software in the world, all you need to do is download (and unzip) -> launch it, there's nothing else required.
</details>

<details>
<summary> Q: What is 'run_anime.bat' used for?</summary>
'run.bat' will enable the generic version of Fooocus-ControlNet-SDXL, while 'run_anime.bat' will start the animated version of Fooocus-ControlNet-SDXL. 
The animated version of Fooocus-ControlNet-SDXL doesn't have any magical spells inside; 
it simply changes some default configurations from the generic version. 
You can try launching both the generic and animated versions separately to see if there are any differences in the user interface.
</details>

<details>
<summary>Q: What is 'run_realistic.bat' used for?</summary>
Realistic version of Fooocus-ControlNet-SDXL
</details>

<details>
<summary>Q: How to speed up ?</summary>
In the first time you launch the software, it will automatically download models(only for <span style="color:green;">online Version</span>):
1. It will download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors)
   as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".
2. It will
   download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors)
   as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors".
3. Note that if you use inpaint, at the first time you inpaint an image, it will
   download [Fooocus's own inpaint control model from here](https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch)
   as the file "Fooocus\models\inpaint\inpaint.fooocus.patch" (the size of this file is 1.28GB).

After Fooocus-ControlNet-SDXL 2.1.60(*for convenience, the version of Fooocus-ControlNet-SDXL follows fooocus*),
you will also have `run_anime.bat` and `run_realistic.bat`.
They are different model presets (and requires different models, but they will be automatically downloaded).
[Check here for more details](https://github.com/lllyasviel/Fooocus/discussions/679).

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

🚀If you already have these files, you can copy them to the above locations to speed up installation.🚀

Note that if you see **"MetadataIncompleteBuffer" or "PytorchStreamReader"**, then your model files are corrupted.
Please download models again.

Below is a test on a relatively low-end laptop with **16GB System RAM** and **6GB VRAM** (Nvidia 3060 laptop). The speed on this machine is about 1.35 seconds per iteration.
Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

For faster rendering speeds and reduced computer lag, consider buy more RAM. This is the most cost-effective solution, as RAM is significantly cheaper than graphics memory (VRAM).
Cost-effective option suggestion: 🚀 32GB of RAM + 8GB of VRAM 🚀or 🚀48GB of RAM + 8GB of VRAM 🚀

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Besides, recently many other software report that Nvidia driver above 532 is sometimes 10x slower than Nvidia driver 531.
If your generation time is very long, consider

* 🚀download [Nvidia Driver 531 Laptop](https://www.nvidia.com/download/driverResults.aspx/199991/en-us/)
* 🚀or [Nvidia Driver 531 Desktop](https://www.nvidia.com/download/driverResults.aspx/199990/en-us/).

Note that the minimal requirement is **4GB Nvidia GPU memory (4GB VRAM)** and **8GB system memory (8GB RAM)**. This
requires using Microsoft’s Virtual Swap technique, which is automatically enabled by your Windows installation in most
cases, so you often do not need to do anything about it. However, if you are not sure, or if you manually turned it
off (would anyone really do that?), or **if you see any "RuntimeError: CPUAllocator"**, you can enable it here:



<details>
<summary>Click here to the see the image instruction. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**And make sure that you have at least 40GB free space on each drive if you still see "RuntimeError: CPUAllocator" !**

</details>

Please open an issue if you use similar devices but still cannot achieve acceptable performances.
</details>
</details>

---

### Colab

<details>
<summary>Click here to the see</summary>
(Last tested - 2023 Oct 10)

| Colab                                                                                                                                                                                         | Info                    |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenneishi/Fooocus-ControlNet-SDXL/blob/main/fooocusControl_colab.ipynb) | FooocusControl Official |

Note that this Colab will disable refiner by default because Colab free's resource is relatively limited.

Thanks to [camenduru](https://github.com/camenduru)!
</details>

---

### Linux (Using Anaconda)

<details>
<summary>Click here to the see</summary>
If you want to use Anaconda/Miniconda, you can

    git clone https://github.com/fenneishi/Fooocus-ControlNet-SDXL.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocusControl
    pip install pygit2==1.12.2

Then download the models:
download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors)
as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors", and
download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors)
as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors". **Or let Fooocus automatically download
the models** using the launcher:

    conda activate fooocusControl
    python entry_with_update.py

Or if you want to open a remote port, use

    conda activate fooocusControl
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus
Anime/Realistic Edition.
</details>

---

### Linux (Using Python Venv)

<details>
<summary>Click here to the see</summary>
Your Linux needs to have **Python 3.10** installed, and lets say your Python can be called with command **python3** with
your venv system working, you can

    git clone https://github.com/fenneishi/Fooocus-ControlNet-SDXL.git
    cd Fooocus
    python3 -m venv fooocus_control_env
    source fooocus_control_env/bin/activate
    pip install pygit2==1.12.2

See the above sections for model downloads. You can launch the software with:

    source fooocus_control_env/bin/activate
    python entry_with_update.py

Or if you want to open a remote port, use

    source fooocus_control_env/bin/activate
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus
Anime/Realistic Edition.
</details>

---

### Linux (Using native system Python)

<details>
<summary>Click here to the see</summary>
If you know what you are doing, and your Linux already has **Python 3.10** installed, and your Python can be called with
command **python3** (and Pip with **pip3**), you can

    git clone https://github.com/fenneishi/Fooocus-ControlNet-SDXL.git
    cd Fooocus
    pip3 install pygit2==1.12.2

See the above sections for model downloads. You can launch the software with:

    python3 entry_with_update.py

Or if you want to open a remote port, use

    python3 entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus
Anime/Realistic Edition.
</details>

---

### Linux (AMD GPUs)

<details>
<summary>Click here to the see</summary>
Same with the above instructions. You need to change torch to AMD version

    pip uninstall torch torchvision torchaudio torchtext functorch xformers 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

AMD is not intensively tested, however. The AMD support is in beta.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus
Anime/Realistic Edition.
</details>

---

### Windows(AMD GPUs)

<details>
<summary>Click here to the see</summary>
Same with Windows. Download the software, edit the content of `run.bat` as:

    .\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
    .\python_embeded\python.exe -m pip install torch-directml
    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
    pause

Then run the `run.bat`.

AMD is not intensively tested, however. The AMD support is in beta.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus
Anime/Realistic Edition.
</details>

---

### Mac

<details>
<summary>Click here to the see</summary>
Mac is not intensively tested. Below is an unofficial guideline for using Mac. You can discuss
problems [here](https://github.com/lllyasviel/Fooocus/pull/129).

You can install Fooocus on Apple Mac silicon (M1 or M2) with macOS 'Catalina' or a newer version. Fooocus runs on Apple
silicon computers via [PyTorch](https://pytorch.org/get-started/locally/) MPS device acceleration. Mac Silicon computers
don't come with a dedicated graphics card, resulting in significantly longer image processing times compared to
computers with dedicated graphics cards.

1. Install the conda package manager and pytorch nightly. Read
   the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide for
   instructions. Make sure pytorch recognizes your MPS device.
1. Open the macOS Terminal app and clone this repository
   with `git clone https://github.com/fenneishi/Fooocus-ControlNet-SDXL.git`.
1. Change to the new Fooocus directory, `cd Fooocus`.
1. Create a new conda environment, `conda env create -f environment.yaml`.
1. Activate your new conda environment, `conda activate fooocusControl`.
1. Install the pygit2, `pip install pygit2==1.12.2`.
1. Install the packages required by Fooocus, `pip install -r requirements_versions.txt`.
1. Launch Fooocus by running `python entry_with_update.py`. The first time you run Fooocus, it will automatically
   download the Stable Diffusion SDXL models and will take a significant time, depending on your internet connection.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus
Anime/Realistic Edition.
</details>

---

# <center>💤Customization your path💤</center>

<details>
<summary>click here to show</summary>


After the first time you run Fooocus, a config file will be generated at `Fooocus\user_path_config.txt`. This file can
be edited for changing the model path. You can also change some parameters to turn Fooocus into "your Fooocus".

For
example ["realisticStockPhoto_v10" is a pretty good model from CivitAI](https://civitai.com/models/139565/realistic-stock-photo).
This model needs a special `CFG=3.0` and probably works better with some specific styles. Below is an example config to
turn Fooocus into a **"Fooocus Realistic Stock Photo Software"**:

`Fooocus\user_path_config.txt`:

```json
{
  "modelfile_path": "D:\\Fooocus\\models\\checkpoints",
  "lorafile_path": "D:\\Fooocus\\models\\loras",
  "vae_approx_path": "D:\\Fooocus\\models\\vae_approx",
  "upscale_models_path": "D:\\Fooocus\\models\\upscale_models",
  "inpaint_models_path": "D:\\Fooocus\\models\\inpaint",
  "controlnet_models_path": "D:\\Fooocus\\models\\controlnet",
  "clip_vision_models_path": "D:\\Fooocus\\models\\clip_vision",
  "fooocus_expansion_path": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
  "temp_outputs_path": "D:\\Fooocus\\outputs",
  "default_model": "realisticStockPhoto_v10.safetensors",
  "default_refiner": "",
  "default_lora": "",
  "default_lora_weight": 0.25,
  "default_cfg_scale": 3.0,
  "default_sampler": "dpmpp_2m",
  "default_scheduler": "karras",
  "default_negative_prompt": "low quality",
  "default_positive_prompt": "",
  "default_styles": [
    "Fooocus V2",
    "Default (Slightly Cinematic)",
    "SAI Photographic"
  ]
}
```

</details>

#

#

#

#

---

# <center>🔥About Fooocus(Thanks to [lllyasviel](https://github.com/lllyasviel/Fooocus) great work! )🔥</center>

#           

<details>
<summary>click here to show </summary>

<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/f79c5981-cf80-4ee3-b06b-3fef3f8bfbc7" width=100%>

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/)).

Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs:

* Learned from Stable Diffusion, the software is offline, open source, and free.

* Learned from Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images.

Fooocus has included and automated [lots of inner optimizations and quality improvements](#tech_list). Users can forget
all those difficult technical parameters, and just enjoy the interaction between human and computer to "explore new
mediums of thought and expanding the imaginative powers of the human species" `[1]`.

Fooocus has simplified the installation. Between pressing "download" and generating the first image, the number of
needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

`[1]` David Holz, 2019.

# Moving from Midjourney to Fooocus

Using Fooocus is as easy as (probably easier than) Midjourney – but this does not mean we lack functionality. Below are
the details.

| Midjourney                                                                                                    | Fooocus                                                                                                                                                                                                                                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Unknown method) | High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Fooocus has offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter your prompt is as short as “house in garden” or as long as 1000 words)                                                          |
| V1 V2 V3 V4                                                                                                   | Input Image -> Upscale or Variation -> Vary (Subtle) / Vary (Strong)                                                                                                                                                                                                                                                                                                     |
| U1 U2 U3 U4                                                                                                   | Input Image -> Upscale or Variation -> Upscale (1.5x) / Upscale (2x)                                                                                                                                                                                                                                                                                                     |
| Inpaint / Up / Down / Left / Right (Pan)                                                                      | Input Image -> Inpaint or Outpaint -> Inpaint / Up / Down / Left / Right <br> (Fooocus uses its own inpaint algorithm and inpaint models so that results are more satisfying than all other software that uses standard SDXL inpaint method/model)                                                                                                                       |
| Image Prompt                                                                                                  | Input Image -> Image Prompt <br> (Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than all other software that uses standard SDXL methods like standard IP-Adapters or Revisions)                                                                                                                        |
| --style                                                                                                       | Advanced -> Style                                                                                                                                                                                                                                                                                                                                                        |
| --stylize                                                                                                     | Advanced -> Advanced -> Guidance                                                                                                                                                                                                                                                                                                                                         |
| --niji                                                                                                        | [Multiple launchers: "run.bat", "run_anime.bat", and "run_realistic.bat".](https://github.com/lllyasviel/Fooocus/discussions/679) <br> Fooocus support SDXL models on Civitai <br> (You can google search “Civitai” if you do not know about it)                                                                                                                         |
| --quality                                                                                                     | Advanced -> Quality                                                                                                                                                                                                                                                                                                                                                      |
| --repeat                                                                                                      | Advanced -> Image Number                                                                                                                                                                                                                                                                                                                                                 |
| Multi Prompts (::)                                                                                            | Just use multiple lines of prompts                                                                                                                                                                                                                                                                                                                                       |
| Prompt Weights                                                                                                | You can use " I am (happy:1.5)". <br> Fooocus uses A1111's reweighting algorithm so that results are better than ComfyUI if users directly copy prompts from Civitai. (Because if prompts are written in ComfyUI's reweighting, users are less likely to copy prompt texts as they prefer dragging files) <br> To use embedding, you can use "(embedding:file_name:1.1)" |
| --no                                                                                                          | Advanced -> Negative Prompt                                                                                                                                                                                                                                                                                                                                              |
| --ar                                                                                                          | Advanced -> Aspect Ratios                                                                                                                                                                                                                                                                                                                                                |

We also have a few things borrowed from the best parts of LeonardoAI:

| LeonardoAI                                                | Fooocus                                          |
|-----------------------------------------------------------|--------------------------------------------------|
| Prompt Magic                                              | Advanced -> Style -> Fooocus V2                  |
| Advanced Sampler Parameters (like Contrast/Sharpness/etc) | Advanced -> Advanced -> Sampling Sharpness / etc |
| User-friendly ControlNets                                 | Input Image -> Image Prompt -> Advanced          |

Fooocus also developed many "fooocus-only" features for advanced users to get perfect
results. [Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

## List of "Hidden" Tricks

<a name="tech_list"></a>

Below things are already inside the software, and **users do not need to do anything about these**.

1.

GPT2-based [prompt expansion as a dynamic style "Fooocus V2".](https://github.com/lllyasviel/Fooocus/discussions/117#raw) (
similar to Midjourney's hidden pre-processsing and "raw" mode, or the LeonardoAI's Prompt Magic).

2. Native refiner swap inside one single k-sampler. The advantage is that now the refiner model can reuse the base
   model's momentum (or ODE's history parameters) collected from k-sampling to achieve more coherent sampling. In
   Automatic1111's high-res fix and ComfyUI's node system, the base model and refiner use two independent k-samplers,
   which means the momentum is largely wasted, and the sampling continuity is broken. Fooocus uses its own advanced
   k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup. (Update Aug 13: Actually
   I discussed this with Automatic1111 several days ago and it seems that the “native refiner swap inside one single
   k-sampler” is [merged]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) into the dev branch of
   webui. Great!)
3. Negative ADM guidance. Because the highest resolution level of XL Base does not have cross attentions, the positive
   and negative signals for XL's highest resolution level cannot receive enough contrasts during the CFG sampling,
   causing the results look a bit plastic or overly smooth in certain cases. Fortunately, since the XL's highest
   resolution level is still conditioned on image aspect ratios (ADM), we can modify the adm on the positive/negative
   side to compensate for the lack of CFG contrast in the highest resolution level. (Update Aug 16, the IOS
   App [Drawing Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) will support Negative ADM
   Guidance. Great!)
4. We implemented a carefully tuned variation of the Section 5.1
   of ["Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"](https://arxiv.org/pdf/2210.00939.pdf).
   The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield overly
   smooth or plastic appearance (examples [here](https://github.com/lllyasviel/Fooocus/discussions/117#sharpness)). This
   can almostly eliminate all cases that XL still occasionally produce overly smooth results even with negative ADM
   guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure
   preservation and fewer artifacts.)
5. We modified the style templates a bit and added the "cinematic-default".
6. We tested the "sd_xl_offset_example-lora_1.0.safetensors" and it seems that when the lora weight is below 0.5, the
   results are always better than XL without lora.
7. The parameters of samplers are carefully tuned.
8. Because XL uses positional encoding for generation resolution, images generated by several fixed resolutions look a
   bit better than that from arbitrary resolutions (because the positional encoding is not very good at handling int
   numbers that are unseen during training). This suggests that the resolutions in UI may be hard coded for best
   results.
9. Separated prompts for two different text encoders seem unnecessary. Separated prompts for base model and refiner may
   work but the effects are random, and we refrain from implement this.
10. DPM family seems well-suited for XL, since XL sometimes generates overly smooth texture but DPM family sometimes
    generate overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.
11. A carefully designed system for balancing multiple styles as well as prompt expansion.
12. Using automatic1111's method to normalize prompt emphasizing. This significantly improve results when users directly
    copy prompts from civitai.
13. The joint swap system of refiner now also support img2img and upscale in a seamless way.
14. CFG Scale and TSNR correction (tuned for SDXL) when CFG is bigger than 10.

## Customization

After the first time you run Fooocus, a config file will be generated at `Fooocus\user_path_config.txt`. This file can
be edited for changing the model path. You can also change some parameters to turn Fooocus into "your Fooocus".

For
example ["realisticStockPhoto_v10" is a pretty good model from CivitAI](https://civitai.com/models/139565/realistic-stock-photo).
This model needs a special `CFG=3.0` and probably works better with some specific styles. Below is an example config to
turn Fooocus into a **"Fooocus Realistic Stock Photo Software"**:

`Fooocus\user_path_config.txt`:

```json
{
  "modelfile_path": "D:\\Fooocus\\models\\checkpoints",
  "lorafile_path": "D:\\Fooocus\\models\\loras",
  "vae_approx_path": "D:\\Fooocus\\models\\vae_approx",
  "upscale_models_path": "D:\\Fooocus\\models\\upscale_models",
  "inpaint_models_path": "D:\\Fooocus\\models\\inpaint",
  "controlnet_models_path": "D:\\Fooocus\\models\\controlnet",
  "clip_vision_models_path": "D:\\Fooocus\\models\\clip_vision",
  "fooocus_expansion_path": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
  "temp_outputs_path": "D:\\Fooocus\\outputs",
  "default_model": "realisticStockPhoto_v10.safetensors",
  "default_refiner": "",
  "default_lora": "",
  "default_lora_weight": 0.25,
  "default_cfg_scale": 3.0,
  "default_sampler": "dpmpp_2m",
  "default_scheduler": "karras",
  "default_negative_prompt": "low quality",
  "default_positive_prompt": "",
  "default_styles": [
    "Fooocus V2",
    "Default (Slightly Cinematic)",
    "SAI Photographic"
  ]
}
```

Consider twice before you really change the config. If you find yourself breaking things, just
delete `Fooocus\user_path_config.txt`. Fooocus will go back to default.

A safter way is just to try "run_anime.bat" or "run_realistic.bat" - they should be already good enough for different
tasks.

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

Fooocus also has many community forks, just like SD-WebUI, for enthusiastic users who want to try!

| SD-WebUI's forks                                                                                                                                                                    | Fooocus' forks                                                                                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [vladmandic/automatic](https://github.com/vladmandic/automatic) </br> [anapnoe/stable-diffusion-webui-ux](https://github.com/anapnoe/stable-diffusion-webui-ux) </br> and so on ... | [runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> and so on ... |

See also [About Forking and Promotion of Forks](https://github.com/lllyasviel/Fooocus/discussions/699).

## Thanks

Fooocus is powered by [FCBH backend](https://github.com/lllyasviel/Fooocus/tree/main/backend), which starts from an odd
mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
and [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

Special thanks to [twri](https://github.com/twri) and [3Diva](https://github.com/3Diva) for creating additional SDXL
styles available in Fooocus.

## Update Log

The log is [here](update_log.md).
</details>

