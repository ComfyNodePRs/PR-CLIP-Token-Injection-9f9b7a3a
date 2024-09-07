import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import os
import re
from math import copysign, pi, sin
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
                "pos_vs_neg":   ("FLOAT",  {"default": 0.5, "min":  0.0, "max": 1.0, "step": 1/10}),
                "strength": ("FLOAT",  {"default": 1.0, "min": -1.0, "max": 1.0, "step": 1/100}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)

    def exec(self, clip, target_text, text_add, text_sub, pos_vs_neg, strength):
        bf = ne(text_add) + ne(text_sub)
        if target_text == "" or bf == 0:
            return clip,

        c = clip.clone()
        pvsn = clamp(pos_vs_neg + ne(text_add) - ne(text_sub), 0, 1)
        s_mul = (2 - abs(0.5 - pvsn) / 0.5)

        def blend_tokens(t):
            if len(t) == 1:
                return t[0]
            if not isinstance(t, torch.Tensor):
                t = torch.stack(t)
            return t.sum(dim=0) / t.norm(dim=1, keepdim=True).sum(dim=0)

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

                if strength < 0:
                    w_add, w_sub = w_sub, w_add
                for t in target_ids:
                    w_pat = weights_patch[t]
                    weights_patch[t] = (w_add * pvsn - w_sub * (1 - pvsn)) * abs(strength * s_mul) + w_pat * (1 - abs(strength))

                c.patcher.add_object_patch(f"clip_{k}.transformer.text_model.embeddings.token_embedding.weight", torch.nn.Parameter(weights_patch.to(default_device)))
        return c,

class levelCLIPWeights:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "level_clip" : (["disabled","enabled","enabled_and_multiply_by_mean_norm"],),
                "subtract_mean" : (["disabled","globally","per_token"],),
                "reset_special" : ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, level_clip, subtract_mean, reset_special):
        c = clip.clone()
        for j, k in enumerate(CLIP_LETTERS):
            weights_patch, _, _ = get_weights_and_tokenizer(c,k)
            ignored_token_weights = []
            if weights_patch is not None:
                weights_patch = weights_patch.clone().to(default_device)
                
                for g in ignored_token_ids:
                    ignored_token_weights.append(weights_patch[g].clone())

                if level_clip != "disabled":
                    norm_w = weights_patch.norm(dim=1, keepdim=True)
                    if level_clip == "enabled_and_multiply_by_mean_norm":
                        weights_patch = weights_patch * (norm_w.mean() + 1e-3) / (norm_w + 1e-3)
                    else:
                        weights_patch = weights_patch / (norm_w + 1e-3)

                if subtract_mean == "globally":
                    weights_patch -= weights_patch.mean()
                elif subtract_mean == "per_token":
                    weights_patch -= weights_patch.mean(dim=1, keepdim=True)

                if reset_special:
                    for j, g in enumerate(ignored_token_ids):
                        weights_patch[g] = ignored_token_weights[j]

                c.patcher.add_object_patch(f"clip_{k}.transformer.text_model.embeddings.token_embedding.weight", torch.nn.Parameter(weights_patch))
        return c,

class turnCLIPWeightsIntoSign:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "turn_into_sign" : ("BOOLEAN", {"default": True, "tooltip":"The weights all becomes 1, 0 or -1\nThe end result is multiplied by the maximum absolute value found in the first token."}),
                "add_1e_3" : ("BOOLEAN", {"default": True, "tooltip":"Add 1e-3 to the weights and they will become 1 or -1 instead\nThe end result is multiplied by the maximum absolute value found in the first token."}),
                # "multiply_by_max_first_token" : ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, turn_into_sign, add_1e_3, multiply_by_max_first_token=True):
        clip = clip.clone()
        for c, k in enumerate(["g","l"]):
            ignored_token_weights = []
            cond_stage_model = getattr(clip.cond_stage_model, f"clip_{k}", None)
            if cond_stage_model is not None:
                weights = cond_stage_model.transformer.text_model.embeddings.token_embedding.weight
                device = weights.device
                cdtype = weights.dtype

                if k not in self.clips_storage:
                    self.clips_storage[k] = torch.clone(weights).to(device="cpu")
                    all_weights = torch.clone(weights).to(device=model_management.get_torch_device())
                else:
                    all_weights = torch.clone(self.clips_storage[k]).to(device=model_management.get_torch_device())

                if turn_into_sign:
                    for g in ignored_token_ids:
                        ignored_token_weights.append(all_weights[g].clone())

                    # backup_nan  = all_weights.clone()

                    max_fist_token = all_weights[0].abs().max().item()
                    print(f"CLIP_{k} max abs value of first token is {max_fist_token}")

                    if add_1e_3:
                        all_weights += 1e-3

                    all_weights = all_weights.sign()

                    zeros_values = (all_weights == 0).sum().item()
                    print(f"total values equal to 0: {zeros_values}")

                    if multiply_by_max_first_token:
                        all_weights = all_weights * max_fist_token

                    # nan_values  = torch.isnan(all_weights)
                    # all_weights[nan_values] = backup_nan[nan_values]

                    for j, g in enumerate(ignored_token_ids):
                        all_weights[g] = ignored_token_weights[j]

                cond_stage_model.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(all_weights.to(device=device,dtype=cdtype))
        return clip,

class replaceCLIPWeights:
    def __init__(self):
        self.clips_storage = {}
        msg = "Only the CLIP to pick from can be changed.\nUSE A NEW NODE OR RESTART THE UI TO EDIT ANOTHER CLIP MODEL\n\
To reset the current model to its normal state turn the toggle on and run a batch before unplugging!\n\
Do not use with another similar node."
        print(f" \033[91m\n{msg}\033[0m")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_to_modify": ("CLIP",),
                "clip_to_pick_from": ("CLIP",),
                "target_text": ("STRING", {"multiline": True}),
                "output_original" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip_to_modify, clip_to_pick_from, target_text, output_original):
        clip_to_modify = clip_to_modify.clone()
        for c, k in enumerate(["g","l"]):
            csm1 = getattr(clip_to_modify.cond_stage_model, f"clip_{k}", None)
            csm2 = getattr(clip_to_pick_from.cond_stage_model, f"clip_{k}", None)
            if csm1 is not None and csm2 is not None:
                tokenizer = getattr(clip_to_pick_from.tokenizer, f"clip_{k}", None)
                tids = words_to_tokens_ids(tokenizer, target_text)
                if c == 0: print(f"Total tokens: {len(tids)}")
                w1 = csm1.transformer.text_model.embeddings.token_embedding.weight
                w2 = csm2.transformer.text_model.embeddings.token_embedding.weight
                device = w1.device
                if k not in self.clips_storage:
                    self.clips_storage[k] = torch.clone(w1).to(device="cpu")
                    new_w = torch.clone(w1).to(device=device)
                else:
                    new_w = torch.clone(self.clips_storage[k]).to(device=device)
                if not output_original and len(tids) > 0:
                    for i in tids:
                        new_w[i] = w2[i].clone()
                csm1.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(new_w)
        return clip_to_modify,

class fixCLIPWeights:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_to_fix": ("CLIP",),
                "clip_doctor": ("CLIP",),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip_to_fix, clip_doctor):
        clip_to_fix = clip_to_fix.clone()
        for c, k in enumerate(["g","l"]):
            csm_fix = getattr(clip_to_fix.cond_stage_model, f"clip_{k}", None)
            csm_doc = getattr(clip_doctor.cond_stage_model, f"clip_{k}", None)
            if csm_fix is not None and csm_doc is not None:
                weights_fix = csm_fix.transformer.text_model.embeddings.token_embedding.weight
                weights_doc = csm_doc.transformer.text_model.embeddings.token_embedding.weight
                bobo = torch.isnan(weights_fix)
                weights_fix[bobo] = weights_doc[bobo]
                csm_fix.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(weights_fix)
        return clip_to_fix,

class saveCustomTokenNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "target_text": ("STRING", {"multiline": True,"tooltip":"Which tokens are to be saved. Commas are ignored."}),
                "filename": ("STRING", {"multiline": False, "default":"filename"}),
                "token_ids_to_include" : ("STRING", {"multiline": False,"default":"", "tooltip":"integers, comma separated."})
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def exec(self, clip, target_text, filename, token_ids_to_include=""):
        to_save = {}
        for c, k in enumerate(["g","l"]):
            csm1 = getattr(clip.cond_stage_model, f"clip_{k}", None)
            tokenizer = getattr(clip.tokenizer, f"clip_{k}", None)
            if csm1 is not None:
                to_save[k] = {}
                tids = words_to_tokens_ids(tokenizer, target_text)
                titi = [int(s) for s in token_ids_to_include.split(",") if s != ""]
                tids = tids + titi
                weights = csm1.transformer.text_model.embeddings.token_embedding.weight
                for t in tids:
                    to_save[k][t] = weights[t].clone().cpu()
        if len(to_save) > 0:
            file_path = os.path.join(weights_output_path, f"{filename}.pt")
            torch.save(to_save,file_path)
            print(f"Weights saved at path: {file_path}")
        else:
            print("The only compatible models are SD1.x/SD2/SDXL. Nothing will be saved.")
        return {}

class loadCustomTokenNode:
    def __init__(self):
        self.original_weights = {}
        self.last_file_loaded = ""

    @classmethod
    def INPUT_TYPES(s):
        wavg = s.IS_WEIGHTED_AVERAGE
        filenames = os.listdir(weights_output_path)
        filenames = sorted(filenames, key=str.lower)
        filenames = [f.replace(".pt","") for f in filenames]
        required  = {
            "clip": ("CLIP",),
            "filename": (filenames,)
            }
        if not wavg:
            required["output_original"] = ("BOOLEAN", {"default": False})
        else:
            required["loaded_weights_strength"] = ("FLOAT", {"default": 1, "min": .0, "max": 1.0, "step": 1/100})
        # required["use_torch_norm"] = ("BOOLEAN", {"default": False})
        return {"required": required}

    IS_WEIGHTED_AVERAGE = False
    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, filename, output_original=True, loaded_weights_strength=0, use_torch_norm=False):
        clip = clip.clone()
        file_path = os.path.join(weights_output_path, f"{filename}.pt")
        custom_weights = torch.load(file_path)
        for k in custom_weights:
            if k not in self.original_weights:
                self.original_weights[k] = {}
            csm1 = getattr(clip.cond_stage_model, f"clip_{k}", None)
            if csm1 is not None:
                clip_weights = csm1.transformer.text_model.embeddings.token_embedding.weight.clone()
                device = clip_weights.device
                for t in custom_weights[k]:
                    if t in self.original_weights[k]:
                        if output_original:
                            clip_weights[t] = self.original_weights[k][t].clone().to(device=device)
                    else:
                        self.original_weights[k][t] = clip_weights[t].clone().to(device='cpu')
                    if loaded_weights_strength != 0:
                        old_weight = self.original_weights[k][t].clone().to(device=device)
                        new_weight = custom_weights[k][t].clone().to(device=device)
                        if use_torch_norm:
                            new_weight = old_weight.norm() * new_weight / new_weight.norm()
                        clip_weights[t] = new_weight * loaded_weights_strength + old_weight * (1 - loaded_weights_strength)
                    if not output_original:
                        clip_weights[t] = custom_weights[k][t].clone().to(device=device)
                csm1.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(clip_weights)
        return clip,

class loadCustomTokenPerpNode:
    def __init__(self):
        self.original_weights = {}

    @classmethod
    def INPUT_TYPES(s):
        filenames = os.listdir(weights_output_path)
        filenames = sorted(filenames, key=str.lower)
        filenames = [f.replace(".pt","") for f in filenames]
        required  = {
            "clip": ("CLIP",),
            "filename": (filenames,)
            }
        required["strength"] = ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 1/100, "tooltip":"The recommanded weight is 0.5"})
        return {"required": required}

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def diff_vec(self, a, b):
        norm_a = torch.linalg.norm(a)
        res = b - a * (a / norm_a * (b / norm_a)).sum()
        return torch.nan_to_num(res, nan=0.0)

    def exec(self, clip, filename, strength=0):
        clip = clip.clone()
        file_path = os.path.join(weights_output_path, f"{filename}.pt")
        custom_weights = torch.load(file_path)
        for k in custom_weights:
            if k not in self.original_weights:
                self.original_weights[k] = {}
            csm1 = getattr(clip.cond_stage_model, f"clip_{k}", None)
            if csm1 is not None:
                clip_weights = csm1.transformer.text_model.embeddings.token_embedding.weight.clone()
                device = clip_weights.device
                for t in custom_weights[k]:
                    if t in self.original_weights[k]:
                        clip_weights[t] = self.original_weights[k][t].clone().to(device=device)
                    else:
                        self.original_weights[k][t] = clip_weights[t].clone().to(device='cpu')
                    if strength != 0:
                        old_weight = self.original_weights[k][t].clone().to(device=device)
                        new_weight = custom_weights[k][t].clone().to(device=device)
                        new_weight_perp = self.diff_vec(old_weight, new_weight)
                        clip_weights[t] = old_weight + (new_weight_perp - old_weight) * strength
                csm1.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(clip_weights)
        return clip,

NODE_CLASS_MAPPINGS = {
    "CLIP token injection": ClipTokenLobotomyNode,
    "CLIP take weights from another": replaceCLIPWeights,
    "CLIP level weights": levelCLIPWeights,
    "CLIP turn weights into sign": turnCLIPWeightsIntoSign,
    "CLIP fix nans (can be chained, best at end)": fixCLIPWeights,
    "CLIP save custom tokens": saveCustomTokenNode,
    "CLIP load custom tokens": type("loadCustomTokenNode", (loadCustomTokenNode,), { "IS_WEIGHTED_AVERAGE": False}),
    "CLIP load custom tokens (weighted average)": type("loadCustomTokenNodeWAvg", (loadCustomTokenNode,), { "IS_WEIGHTED_AVERAGE": True}),
    "CLIP load custom tokens (perp)": loadCustomTokenPerpNode,
}