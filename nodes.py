import torch
import comfy.model_management as model_management
import os
import re
from math import copysign

sign  = lambda x: copysign(1, x)
clamp = lambda x,vmin,vmax: max(min(x,vmax),vmin)
fnorm = lambda x, y = True: x / torch.norm(x) if y else x
ne    = lambda x: x != ""
ignored_token_ids = [49406, 49407, 0]
default_device = model_management.get_torch_device()
current_dir = os.path.dirname(os.path.realpath(__file__))
weights_output_path = os.path.join(current_dir, "custom_weights")
CLIP_LETTERS = ["g","l"]

def words_to_tokens_ids(clip_model, text_input, allow_double=False):
    if text_input == "":
        return []
    text_input = text_input.replace(" , "," ").replace(" ,"," ").replace(", "," ").replace(","," ").replace("\n"," ")
    text_tokenized = clip_model.tokenizer.tokenize(text_input)
    text_token_ids = clip_model.tokenizer.convert_tokens_to_ids(text_tokenized)
    text_token_ids = [tti for tti in text_token_ids if tti not in ignored_token_ids]
    if allow_double:
        return text_token_ids
    return list(dict.fromkeys(text_token_ids))

def split_or(string):
    delimiters = [" ", ",", "\n"]
    pattern = '|'.join(map(re.escape, delimiters))
    return [part.replace("_"," ") for part in re.split(f'(?:{pattern})+', string) if part]

def get_weights_and_tokenizer(c, k):
    key_patches = c.get_key_patches()
    original_weights = key_patches.get(f"clip_{k}.transformer.text_model.embeddings.token_embedding.weight", [None])[-1]
    weights_patch = c.patcher.object_patches.get(f"clip_{k}.transformer.text_model.embeddings.token_embedding.weight", original_weights)
    tokenizer = getattr(c.tokenizer, f"clip_{k}", None)
    return weights_patch, original_weights, tokenizer

tooltips = {
    "pos_vs_neg":"Determins the influence of the positive vs the negative vectors in the final result.\nAt 0 will only subtract what is in the negative.\nAt 1 will only add what is in the positive.\nIf any of the text inputs is empty this slider becomes useless.\nGenerally better above 0.5",
    "strength":"-1 will reverse positive vs negative\n 0 is disabled\n 1 fully replaces the targeted tokens by the result"
}

class ClipTokenLobotomyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "target_text":  ("STRING", {"multiline": True}),
                "text_add":     ("STRING", {"multiline": True}),
                "text_sub":     ("STRING", {"multiline": True}),
                "pos_vs_neg":   ("FLOAT",  {"default": 0.5, "min":  0.0, "max": 1.0, "step": 1/10, "tooltip":tooltips["pos_vs_neg"]}),
                "strength":     ("FLOAT",  {"default": 1.0, "min": -1.0, "max": 1.0, "step": 1/10, "tooltip":tooltips["strength"]}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/token surgery"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)

    def exec(self, clip, target_text, text_add, text_sub, pos_vs_neg, strength):
        bf = ne(text_add) + ne(text_sub)
        if target_text == "" or bf == 0:
            return clip,

        c = clip.clone()
        # strength = sign(strength) * abs(strength) ** (1 / 2)
        pvsn = clamp(pos_vs_neg + ne(text_add) - ne(text_sub), 0, 1)
        s_mul = (2 - abs(0.5 - pvsn) / 0.5)

        def blend_tokens(t):
            if len(t) == 1:
                return t[0]
            if not isinstance(t, torch.Tensor):
                t = torch.stack(t)
            tns = t.norm(dim=1, keepdim=True)
            return t.mul(tns).sum(dim=0).div(tns.sum(dim=0))

        def get_tokens_for_change(text,c,w,j):
            if text == "": return torch.zeros_like(w[0]).to(default_device)
            individual_words = split_or(text)
            change_tokens = []
            for d in individual_words:
                tfc  = words_to_tokens_ids(c, d, allow_double=True)
                toks = [w[t].clone().to(default_device) for t in tfc]
                toks = blend_tokens(toks)
                change_tokens.append(toks)
            if j == 0: print(f"Tokens detected: {len(change_tokens)}")
            return blend_tokens(change_tokens)

        for j, k in enumerate(CLIP_LETTERS):
            weights_patch, original_weights, tokenizer = get_weights_and_tokenizer(c, k)

            if None not in [original_weights, tokenizer]:
                weights_patch = weights_patch.clone().to(default_device)

                target_ids = words_to_tokens_ids(tokenizer, target_text)
                w_add = get_tokens_for_change(text_add,tokenizer, original_weights, j)
                w_sub = get_tokens_for_change(text_sub,tokenizer, original_weights, j)

                for t in target_ids:
                    w_pat = weights_patch[t]
                    weights_patch[t] = (w_add * pvsn - w_sub * (1 - pvsn)) * (strength * s_mul) + w_pat * (1 - abs(strength))

                c.patcher.add_object_patch(f"clip_{k}.transformer.text_model.embeddings.token_embedding.weight", torch.nn.Parameter(weights_patch.to(default_device)))
        return c,

class saveCustomTokenNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "target_text": ("STRING", {"multiline": True,"tooltip":"Which tokens are to be saved. Commas are ignored."}),
                "filename": ("STRING", {"multiline": False, "default":"filename"}),
                "token_ids_to_include" : ("STRING", {"multiline": False,"default":"", "tooltip":"integers, comma separated."})
            },"optional": {
                "clip": ("CLIP",),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/token surgery"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def exec(self, target_text, filename, token_ids_to_include="", clip=None):
        if clip is None:
            return {}
        to_save = {}
        for j, k in enumerate(["g","l"]):
            weights, _, tokenizer = get_weights_and_tokenizer(clip,k)
            if weights is not None:
                to_save[k] = {}
                tids = words_to_tokens_ids(tokenizer, target_text)
                titi = [int(s) for s in token_ids_to_include.split(",") if s.strip() != ""]
                tids = tids + titi
                for t in tids:
                    to_save[k][t] = weights[t].clone().cpu()
        if len(to_save) > 0:
            file_path = os.path.join(weights_output_path, f"{filename}.pt")
            torch.save(to_save,file_path)
            print(f"Weights saved at path: {file_path}")
        else:
            print("The only compatible models are CLIP G/L. Nothing will be saved.")
        return {}

class loadCustomTokenNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        filenames = os.listdir(weights_output_path)
        filenames = sorted(filenames, key=str.lower)
        filenames = [f.replace(".pt","") for f in filenames]
        required  = {
            "clip": ("CLIP",),
            "filename": (filenames,)
            }
        required["strength"] = ("FLOAT", {"default": 1, "min": .0, "max": 1.0, "step": 1/10})
        return {"required": required}

    FUNCTION = "exec"
    CATEGORY = "advanced/token surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, filename, strength):
        c = clip.clone()
        file_path = os.path.join(weights_output_path, f"{filename}.pt")
        custom_weights = torch.load(file_path)
        for k in custom_weights:
            weights_patch, original_weights, _ = get_weights_and_tokenizer(c, k)
            if original_weights is not None:
                for t in custom_weights[k]:
                    weights_patch[t] = custom_weights[k][t] * strength + original_weights[t] * (1 - strength)
                c.patcher.add_object_patch(f"clip_{k}.transformer.text_model.embeddings.token_embedding.weight", torch.nn.Parameter(weights_patch.to(default_device)))
        return c,

NODE_CLASS_MAPPINGS = {
    "CLIP token injection": ClipTokenLobotomyNode,
    "CLIP save custom tokens": saveCustomTokenNode,
    "CLIP load custom tokens": loadCustomTokenNode,
}
