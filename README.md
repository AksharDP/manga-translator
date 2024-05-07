# manga-translator

A manga translator that takes in images and translates the speech bubbles to english
<hr/>
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

You can also change the language detection for PaddleOCR in config.ini

```mermaid
graph TD;
    A[Image]-->B[Speech bubble segmentation model];
    B[Speech bubble segmentation model]-->C[OCR all text within segmnetation];
    C[OCR all text within segmnetation]-->D[Dranslate text];
    D[Translate text]-->E[Generate text as image]
    E[Generate text as image]-->F[Overlap image on to original]
```

<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/resource.jpg">
<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/translated_resource.jpg">

<br/>

<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/resource_2.jpg">
<img src="https://raw.githubusercontent.com/AksharDP/manga-translator/main/images/translated_resource_2.jpg">