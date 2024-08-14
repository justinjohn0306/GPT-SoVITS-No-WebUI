import os
import sys
import re
import logging
import torch
import soundfile as sf
import numpy as np
import librosa
from time import time as ttime
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Add GPT_SoVITS to the path
sys.path.append(os.path.join(os.getcwd(), "GPT_SoVITS"))

# Importing necessary modules
import LangSegment
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto
from text import chinese

# Additional Imports and GPU Info

# Setup logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

# Define version and pretrained models
version = os.environ.get("version", "v2")
pretrained_sovits_name = [
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", 
    "GPT_SoVITS/pretrained_models/s2G488k.pth"
]
pretrained_gpt_name = [
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", 
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
]

_ = [[], []]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name, pretrained_sovits_name = _

# Define language dictionaries for v1 and v2
dict_language_v1 = {
    "chinese": "all_zh",
    "english": "en",
    "japanese": "all_ja",
    "chinese+english": "zh",
    "japanese+english": "ja",
    "automatic": "auto",
}
dict_language_v2 = {
    "chinese": "all_zh",
    "english": "en",
    "japanese": "all_ja",
    "chinese+english": "zh",
    "japanese+english": "ja",
    "cantonese": "all_yue",
    "korean": "all_ko",
    "cantonese+english": "yue",
    "korean+english": "ko",
    "automatic": "auto",
    "automatic(cantonese)": "auto_yue",
}

# Define punctuation set
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

# Paths for cnhubert and bert models
cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

# Initialize transformers and utility classes
i18n = I18nAuto()

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure feature_extractor module is imported
try:
    from feature_extractor import cnhubert
except ImportError:
    print("Make sure the feature_extractor module is in the GPT_SoVITS directory.")
    sys.exit(1)

cnhubert.cnhubert_base_path = cnhubert_base_path

# Define global variables for loaded model paths
loaded_gpt_path = None
loaded_sovits_path = None

# Function to get BERT feature
def get_bert_feature(text, word2ph):
    global tokenizer, bert_model
    if not "tokenizer" in globals() or not "bert_model" in globals():
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half:
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

# Class to convert dictionary to attributes recursively
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# Function to change SoVITS weights
def change_sovits_weights(sovits_path):
    global vq_model, hps, version, dict_language
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if "pretrained" not in sovits_path:
        del vq_model.enc_q
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)
    dict_language = dict_language_v1 if version == 'v1' else dict_language_v2

# Function to change GPT weights
def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f:
        f.write(gpt_path)

# Function to get spectrogram
def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

# Function to clean text
def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

dtype = torch.float16 if is_half else torch.float32

# Function to get BERT inference
def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)

    return bert

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

# Function to get the first segment of text before punctuation
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

# Function to get phones and BERT embeddings
def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # Since Chinese, Japanese and Korean characters cannot be distinguished, the user input shall prevail.
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text

# Function to merge short texts
def merge_short_text_in_array(texts, threshold):
    if len(texts) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

cache = {}

# Main function for TTS synthesis
def get_tts_wav(ref_wav_paths, prompt_text, prompt_language, text, text_language, how_to_cut="No slice", top_k=20, top_p=0.6, temperature=0.6, ref_free=False, speed=1.0, if_freeze=False):
    global cache

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print("Actual reference text:", prompt_text)

    text = text.strip("\n")
    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if text_language != "en" else "." + text

    print("Actual target text:", text)
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half else np.float32)

    t1 = ttime()

    if not ref_free:
        refers = []
        with torch.no_grad():
            for ref_wav_path in ref_wav_paths:
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                    raise OSError("Each reference audio must be within 3~10 seconds.")
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                wav16k = wav16k.half().to(device) if is_half else wav16k.to(device)
                zero_wav_torch = zero_wav_torch.half().to(device) if is_half else zero_wav_torch.to(device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                codes = vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                refers.append(prompt_semantic.unsqueeze(0).to(device))

        prompt = torch.mean(torch.stack(refers), dim=0)

    t2 = ttime()

    # Slice the text based on the provided method
    if how_to_cut == "Slice once every 4 sentences":
        text = cut1(text)
    elif how_to_cut == "Cut per 50 characters":
        text = cut2(text)
    elif how_to_cut == "Slice by Chinese punct":
        text = cut3(text)
    elif how_to_cut == "Slice by English punct":
        text = cut4(text)
    elif how_to_cut == "Slice by every punct":
        text = cut5(text)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    print("Actual target text (after slicing):", text)
    texts = text.split("\n")
    texts = process_text(texts)  # Filter out invalid text entries
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []

    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    t3_start = ttime()

    for i_text, text in enumerate(texts):
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        print("Actual target text (per sentence):", text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        print("Processed text (per sentence):", norm_text2)

        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        if i_text in cache and if_freeze:
            pred_semantic = cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic

        refer_spectrograms = [get_spepc(hps, ref_wav_path).to(dtype).to(device) for ref_wav_path in ref_wav_paths]
        audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer_spectrograms, speed=speed).detach().cpu().numpy()[0, 0]
        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

    t3_end = ttime()

    t4_start = ttime()

    # Concatenate audio options
    concatenated_audio = np.concatenate(audio_opt, 0) * 32768

    t4_end = ttime()

    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3_end - t3_start, t4_end - t4_start))
    return (t1 - t0, t2 - t1, t3_end - t3_start, t4_end - t4_start, t4_end - t4_start), hps.data.sampling_rate, concatenated_audio.astype(np.int16)

# Function to split text based on punctuation
def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # There will always be punctuation at the end, so just exit, the last segment was already added last time
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

# Functions to cut text in various ways
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

# Contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

# Function to process and filter text entries
def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("Please enter valid text"))
    for text in texts:
        if text not in [None, " ", "", "\n"]:
            _text.append(text)
    return _text

# Main function for inference
def gptsovits_inference(gpt_model_path, sovits_model_path, ref_audio_paths, prompt_text, prompt_language, text, text_language, how_to_cut="No slice", top_k=20, top_p=0.6, temperature=0.6, ref_free=False, speed=1.0, output_file_path=None):
    global loaded_gpt_path, loaded_sovits_path
    if not ref_audio_paths:
        print("You must input at least one reference audio path!")
        return None
    gpt_path = gpt_model_path if gpt_model_path else pretrained_gpt_name[0]
    sovits_path = sovits_model_path if sovits_model_path else pretrained_sovits_name[0]

    if loaded_gpt_path != gpt_path:
        change_gpt_weights(gpt_path)
        loaded_gpt_path = gpt_path
    if loaded_sovits_path != sovits_path:
        change_sovits_weights(sovits_path)
        loaded_sovits_path = sovits_path

    current_output_path = output_file_path

    if not current_output_path:
        opt_root = os.path.join(os.getcwd(), "output")
        os.makedirs(opt_root, exist_ok=True)
        output_count = 1

        while True:
            opt_filename = f"{output_count}_GPTSoVITS.wav"
            current_output_path = os.path.join(opt_root, opt_filename)
            if not os.path.exists(current_output_path):
                break
            output_count += 1

    times, sr, audio_data = get_tts_wav(ref_audio_paths, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed)
    
    sf.write(current_output_path, audio_data, sr, format="wav")
    print("Times:\nref_audio: {:.2f}s text: {:.2f}s phonemes & bert: {:.2f}s gpt: {:.2f}s sovits: {:.2f}s".format(*times))

    return current_output_path