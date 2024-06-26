# Manga Translator

A manga translator that takes in images and translates the speech bubbles to english
<hr/>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Segmentation%20Model-yellow)](https://huggingface.co/AksharPatel/manga-speech-bubble-segmentation)

</div>
## Installation

PaddleOCR requires python 3.9 or below. For this we are going to use conda to create an environment

```bash
git clone https://github.com/AksharDP/manga-translator.git
```

```bash
cd manga-translator
```

```bash
conda create -n "myenv" python=3.9.12
```

```bash
conda activate myenv
```

```bash
pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
pip install -r ./requirements.txt
```

```bash
python main.py
```
Put all the images that you would like to translate into the input folder that is created when you run the program the first time.

You can change the language detection for PaddleOCR in config.ini

There are many ways to improve the program:

* Train segmentation model to be more accurate, smaller, and faster. (Quantization, Threading, deploying via ONNX)
* Train PaddleOCR for each language for better word detection (especially for vertical text)
* Use a model or paid service for more accurate translations (maybe gpt 4 for languages that require context?)
* Add detection for color of speech bubble and text for better looking speech bubbles
* Change font and font size based on text (currently its the same font size and font)


```mermaid
graph TD;
    A[Image]-->B[Generate segmentation area for each speech bubble];
    B[Generate segmentation area for each speech bubble]-->C[OCR all text within segmentation];
    C[OCR all text within segmentation]-->D[Translate text];
    D[Translate text]-->E[Generate text as image]
    E[Generate text as image]-->F[Overlap image on to original]
```

<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/resource.jpg" height="400" width="300"> <img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/translated_resource.jpg" height="400" width="300">

<br/>

<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/resource_2.jpg" height="400" width="300"> <img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/translated_resource_2.jpg" height="400" width="300">

<br/>


<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/resource_3.jpg" height="400" width="300"> <img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/translated_resource_3.jpg" height="400" width="300">