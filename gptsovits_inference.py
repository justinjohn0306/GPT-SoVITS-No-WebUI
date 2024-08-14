import os, sys, re, logging
sys.path.append(os.path.join(os.getcwd(), "GPT_SoVITS"))
import LangSegment
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import soundfile as sf

dict_language = {
    "chinese": "all_zh",#全部按中文识别
    "english": "en",#全部按英文识别#######不变
    "japanese": "all_ja",#全部按日文识别
    "chinese+english": "zh",#按中英混合识别####不变
    "japanese+english": "ja",#按日英混合识别####不变
    "automatic": "auto",#多语种启动切分识别语种
}

models_dir_path = "gptsovits_models"
pretrained_gpt_path = "gptsovits_models/pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
pretrained_sovits_path = "gptsovits_models/pretrained/s2G488k.pth"

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa

from feature_extractor import cnhubert
cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_bert_feature(text, word2ph):
    global tokenizer, bert_model
    
    if not "tokenizer" in globals() or not "bert_model" in globals():
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half == True:
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

def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    print(f'Phonemes for "{text}":\n{phones}')
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
        
    return phones, bert.to(dtype), norm_text

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
    t0 = ttime()
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(f"Reference text: {prompt_text}")
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print(f"Full text: {text}")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
    t1 = ttime()

    if (how_to_cut == "Slice once every 4 sentences"):
        text = cut1(text)
    elif (how_to_cut == "Cut per 50 characters"):
        text = cut2(text)
    elif (how_to_cut == "Slice by Chinese punct"):
        text = cut3(text)
    elif (how_to_cut == "Slice by English punct"):
        text = cut4(text)
    elif (how_to_cut == "Slice by every punct"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(f"Text after slicing:\n{text}")
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    
    t2 = ttime()
    t3, t4, t5 = 0, 0, 0
    
    for text in texts:
        t3s = ttime()
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(f"Text being generated: {text}")
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        print(f"Text after frontend processing: {norm_text2}")
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = None if ref_free else prompt_semantic.unsqueeze(0).to(device)
        t3e = ttime()
        t3 += t3e - t3s
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t4e = ttime()
        t4 += t4e - t3e
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, ref_wav_path) if not ref_free else None
        if refer is not None:
            if is_half == True:
                refer = refer.half().to(device)
            else:
                refer = refer.to(device)
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
                .detach()
                .cpu()
                .numpy()[0, 0]
        )
        max_audio = np.abs(audio).max()
        if max_audio > 1: audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t5e = ttime()
        t5 += t5e - t4e
    return (
        (t1 - t0, t2 - t1, t3, t4, t5),
        hps.data.sampling_rate,
        (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    )

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

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
    return "\n".join(opts)

def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])

def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

def cut5(inp):
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt

def get_gpt_model(model_name, use_pretrained):
    if model_name and not use_pretrained:
        model_dir_path = os.path.join(models_dir_path, model_name)
        for r, _, f_list in os.walk(model_dir_path):
            for f in f_list:
                if f.endswith(".ckpt"):
                    return os.path.join(r, f)
    print(f"Using pretrained GPT model ({pretrained_gpt_path}).")
    return pretrained_gpt_path

def get_sovits_model(model_name, use_pretrained):
    if model_name and not use_pretrained:
        model_dir_path = os.path.join(models_dir_path, model_name)
        for r, _, f_list in os.walk(model_dir_path):
            for f in f_list:
                if f.endswith(".pth"):
                    return os.path.join(r, f)
    print(f"Using pretrained SoVITS model ({pretrained_sovits_path}).")
    return pretrained_sovits_path

loaded_gpt_path = None
loaded_sovits_path = None

def gptsovits_inference(model_name, use_pretrained_gpt, use_pretrained_sovits, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="No slice", top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
    global loaded_gpt_path, loaded_sovits_path
    
    # Skip the reference audio check if ref_free is True
    if not ref_free and not os.path.exists(ref_wav_path):
        print("You must input a reference audio path!")
        return None

    gpt_path = get_gpt_model(model_name, use_pretrained_gpt)
    sovits_path = get_sovits_model(model_name, use_pretrained_sovits)
    
    if loaded_gpt_path != gpt_path:
        change_gpt_weights(gpt_path)
        loaded_gpt_path = gpt_path
    
    if loaded_sovits_path != sovits_path:
        change_sovits_weights(sovits_path)
        loaded_sovits_path = sovits_path

    opt_root = os.path.join(os.getcwd(), "output")
    os.makedirs(opt_root, exist_ok=True)
    output_count = 1

    while True:
        opt_filename = f"{model_name}_GPTSoVITS_{output_count}.wav"
        current_output_path = os.path.join(opt_root, opt_filename)
        if not os.path.exists(current_output_path):
            break
        output_count += 1

    times, sr, audio_data = get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free)
    
    sf.write(
        current_output_path,
        audio_data,
        sr,
        format="wav"
    )
    print("Times:\nref_audio: {:.2f}s text: {:.2f}s phonemes & bert: {:.2f}s gpt: {:.2f}s sovits: {:.2f}s".format(*times))
    return current_output_path
