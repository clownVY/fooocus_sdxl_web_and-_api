import os
import re
import random
import json

from modules.util import get_files_from_folder

# cannot use modules.path - validators causing circular imports
styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
wildcards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../wildcards/'))


def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


styles = {}

styles_files = get_files_from_folder(styles_path, ['.json'])

for x in ['sdxl_styles_fooocus.json',
          'sdxl_styles_sai.json',
          'sdxl_styles_mre.json',
          'sdxl_styles_twri.json',
          'sdxl_styles_diva.json']:
    if x in styles_files:
        styles_files.remove(x)
        styles_files.append(x)

for styles_file in styles_files:
    try:
        with open(os.path.join(styles_path, styles_file), encoding='utf-8') as f:
            for entry in json.load(f):
                name, prompt, negative_prompt = normalize_key(entry['name']), entry['prompt'], entry['negative_prompt']
                styles[name] = (prompt, negative_prompt)
    except Exception as e:
        print(str(e))
        print(f'Failed to load style file {styles_file}')

style_keys = list(styles.keys())
fooocus_expansion = "Fooocus V2"
legal_style_names = [fooocus_expansion] + style_keys

SD_XL_BASE_RATIOS = {
    "0.6": (768, 1280),
    "0.94": (960, 1024),
    "1.07": (1024, 960),
    "1.0": (1024, 1024),
    "1.14": (1024, 1536),
    "1.80": (1080, 600),
    "1.69": (1080, 640),
    "1.50": (1080, 720),
    "1.01": (1080, 1080),
    "1.09": (1080, 1200),
    "0.06": (1080, 1920),
    "1.78": (1920, 1080),
}

aspect_ratios = {}

# import math

for k, (w, h) in SD_XL_BASE_RATIOS.items():
    # txt = f'{w}×{h}'
    txt = f'{w}*{h}'

    # gcd = math.gcd(w, h)
    # txt += f' {w//gcd}:{h//gcd}'

    aspect_ratios[txt] = (w, h)


def apply_style(style, positive):
    p, n = styles[style]
    return p.replace('{prompt}', positive), n


def apply_wildcards(wildcard_text, seed=None, directory=wildcards_path):
    placeholders = re.findall(r'__(\w+)__', wildcard_text)
    if len(placeholders) == 0:
        return wildcard_text

    rng = random.Random(seed)
    for placeholder in placeholders:
        try:
            words = open(os.path.join(directory, f'{placeholder}.txt'), encoding='utf-8').read().splitlines()
            words = [x for x in words if x != '']
            wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
        except IOError:
            print(f'Error: could not open wildcard file {placeholder}.txt, using as normal word.')
            wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)

    return wildcard_text