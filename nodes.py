import torch
import comfy.model_management as model_management
import os
mean = lambda *x: sum(x)/len(x)
ignored_token_ids = [49406, 49407, 0]
default_device = model_management.get_torch_device()
current_dir = os.path.dirname(os.path.realpath(__file__))
weights_output_path = os.path.join(current_dir, "custom_weights")
# del base_clip,base_model
# comfy.model_management.cleanup_models()

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

class ClipTokenLobotomyNode:
    def __init__(self):
        self.clips_storage = {}
        msg = "WORK IN PROGRESS :p\nTHE NODE IS TIED TO THE MODEL UNTIL RESTART. It keeps the weights in its memory so you can test and edit. Save the result and restart once your done.\n\
To reset the current model to its normal state set the multiplier to 0 and run a batch before unplugging!\n\
Do not chain with another similar node."
        print(f" \033[91m\n{msg}\033[0m")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "target_text": ("STRING", {"multiline": True}),
                
                "text_add": ("STRING", {"multiline": True}),
                "strength_add": ("FLOAT", {"default": 0.5, "min": .0, "max": 1.0, "step": 0.1}),
                "text_sub": ("STRING", {"multiline": True}),
                "strength_sub": ("FLOAT", {"default": 0.5, "min": .0, "max": 1.0, "step": 0.1}),
                "both_strengths_multiplier": ("FLOAT", {"default": 0.5, "min": 0, "max": 100.0, "step": 0.5}),
                "allow_doubles" : ("BOOLEAN", {"default": False, "tooltip":"Allow tokens mentionned multiple times to accumulate.\nThe target tokens remain unique."}),
                "auto_weights" : ("BOOLEAN", {"default": False, "tooltip":"Rebalance the average."}),
                # "apply_to_all_weights" : ("BOOLEAN", {"default": False, "tooltip":"This is a bad idea."}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)

    # def diff_vec(self, a, b):
    #     b = b.unsqueeze(0).repeat(a.size(0) // b.unsqueeze(0).size(0), 1)
    #     norm_a = torch.linalg.norm(a, dim=1).unsqueeze(1)
    #     res = b - a * (a / norm_a * (b / (norm_a + 1e-16))).sum(dim=1).unsqueeze(1)
    #     return torch.nan_to_num(res, nan=0.0)

    def get_weights_from_ids(self,weights,ids):
        warr = {}
        for i in ids:
            warr[i] = weights[i].clone()
        return warr

    def exec(self, clip, target_text, text_add, strength_add, text_sub,
             strength_sub, both_strengths_multiplier, allow_doubles, auto_weights=False):
        clip = clip.clone()
        for c, k in enumerate(["g","l"]):
            weights_pos = {}
            weights_neg = {}
            weight_difs = {}
            ids_pos = []
            ids_neg = []
            clip_tokenizer = getattr(clip.tokenizer, f"clip_{k}", None)
            cond_stage_model = getattr(clip.cond_stage_model, f"clip_{k}", None)

            if cond_stage_model is not None:
                weights = cond_stage_model.transformer.text_model.embeddings.token_embedding.weight
                device = weights.device
                cdtype = weights.dtype

                target_ids = words_to_tokens_ids(clip_tokenizer,target_text)
                ids_pos = words_to_tokens_ids(clip_tokenizer,text_add,allow_doubles)
                ids_neg = words_to_tokens_ids(clip_tokenizer,text_sub,allow_doubles)

                n_pos = len(ids_pos)
                n_neg = len(ids_neg)
                do_pos = (len(text_add) * n_pos * strength_add * both_strengths_multiplier) > 0
                do_neg = (len(text_sub) * n_neg * strength_sub * both_strengths_multiplier) > 0
                do_any = do_pos or do_neg

                if c == 0:
                    print(f"Positive tokens: {n_pos}")
                    print(f"Negative tokens: {n_neg}")

                if k not in self.clips_storage:
                    self.clips_storage[k] = torch.clone(weights).to(device="cpu")
                    all_weights = torch.clone(weights).to(device=model_management.get_torch_device())
                else:
                    all_weights = torch.clone(self.clips_storage[k]).to(device=model_management.get_torch_device())

                if do_any:
                    # target_weigths = self.get_weights_from_ids(all_weights,target_ids)
                    if do_pos:
                        weights_pos = self.get_weights_from_ids(all_weights,ids_pos)
                    if do_neg:
                        weights_neg = self.get_weights_from_ids(all_weights,ids_neg)
                    for t in target_ids:
                        p_res = torch.zeros_like(all_weights[t])
                        n_res = torch.zeros_like(all_weights[t])
                        p_div = 0
                        n_div = 0
                        for i in list(dict.fromkeys([*ids_pos,*ids_neg])):
                            if t == i: continue
                            if i in weights_pos:
                                p_res += (weights_pos[i] - all_weights[t]) * strength_add
                                p_div += 1
                            if i in weights_neg:
                                n_res += (weights_neg[i] - all_weights[t]) * strength_sub
                                n_div += 1
                        p_res = p_res / max(p_div,1)
                        n_res = n_res / max(n_div,1)
                        
                        balance_mult = 1 if not auto_weights else max(1, p_div * strength_add - n_div * strength_sub) / 4
                        if t in weight_difs:
                            weight_difs[t] += (p_res - n_res) * both_strengths_multiplier * balance_mult
                        else:
                            weight_difs[t]  = (p_res - n_res) * both_strengths_multiplier * balance_mult
                    for t in target_ids:
                        all_weights[t] = all_weights[t] + weight_difs[t]
                cond_stage_model.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(all_weights.to(device=device,dtype=cdtype))
        return clip,

class ClipTokenLobotomyAlternativeVersionNode:
    def __init__(self):
        self.clips_storage = {}
        msg = "USE A NEW NODE OR RESTART THE UI TO EDIT ANOTHER CLIP MODEL\n\
To reset the current model to its normal state set the multiplier to 0 and run a batch before unplugging!\n\
Do not use with another similar node."
        print(f" \033[91m\n{msg}\033[0m")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "target_text": ("STRING", {"multiline": True}),
                
                "text_add": ("STRING", {"multiline": True}),
                "strength_add": ("FLOAT", {"default": 0.5, "min": .0, "max": 10.0, "step": 0.5}),
                "text_subtract": ("STRING", {"multiline": True}),
                "strength_subtract": ("FLOAT", {"default": 0.5, "min": .0, "max": 10.0, "step": 0.5}),
                "both_strengths_multiplier": ("FLOAT", {"default": 0.5, "min": 0, "max": 100.0, "step": 0.5}),
                "apply_to_all_weights" : ("BOOLEAN", {"default": False, "tooltip":"This is a bad idea."}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def diff_vec(self, a, b):
        b = b.unsqueeze(0).repeat(a.size(0) // b.unsqueeze(0).size(0), 1)
        norm_a = torch.linalg.norm(a,dim=1).unsqueeze(1)
        res = b - a * (a / norm_a * (b / norm_a)).sum(dim=1).unsqueeze(1)
        return torch.nan_to_num(res, nan=0.0)

    def words_to_tokens_ids(self, clip_model, text_input):
        text_input = text_input.replace(" , "," ").replace(" ,"," ").replace(", "," ").replace(","," ").replace("\n"," ")
        text_tokenized = clip_model.tokenizer.tokenize(text_input)
        text_token_ids = clip_model.tokenizer.convert_tokens_to_ids(text_tokenized)
        text_token_ids = [tti for tti in text_token_ids if tti not in ignored_token_ids]
        return list(dict.fromkeys(text_token_ids))

    def get_weights(self,all_weights,ids):
        nt = len(ids)
        res = torch.zeros_like(all_weights,device=all_weights.device)
        for i in range(nt):
            res += self.diff_vec(all_weights,all_weights[i].clone()) / nt
        return res

    def apply_weights(self,all_weights,weights_diff,target_ids,apply_to_all_weights):
        if apply_to_all_weights:
            # orig_igt = {}
            # for igt in ignored_token_ids:
            #     orig_igt[igt] = all_weights[igt].clone()
            all_weights = all_weights + weights_diff
            # for igt in orig_igt:
            #     all_weights[igt] = orig_igt[igt]
        else:
            for tid in target_ids:
                all_weights[tid] = all_weights[tid] + weights_diff[tid]
        return all_weights

    def pos_neg_weights(self,all_weights,ids_tok,n_tok,strength,target_ids,apply_to_all_weights):
        if n_tok > 0 and strength > 0:
            weights_to_sub = self.get_weights(all_weights, ids_tok)
            all_weights = self.apply_weights(all_weights,weights_to_sub * strength,target_ids,apply_to_all_weights)
        return all_weights

    def exec(self, clip, target_text, text_add, strength_add, text_subtract, strength_subtract, both_strengths_multiplier, apply_to_all_weights):
        clip = clip.clone()

        for c, k in enumerate(["g","l"]):
            clip_tokenizer = getattr(clip.tokenizer, f"clip_{k}", None)
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

                target_ids = self.words_to_tokens_ids(clip_tokenizer,target_text)
                ids_pos = self.words_to_tokens_ids(clip_tokenizer,text_add)
                ids_neg = self.words_to_tokens_ids(clip_tokenizer,text_subtract)

                n_pos = len(ids_pos)
                n_neg = len(ids_neg)

                if c == 0:
                    print(f"Positive tokens: {n_pos}")
                    print(f"Negative tokens: {n_neg}")

                if both_strengths_multiplier > 0:
                    all_weights_pos = self.pos_neg_weights(all_weights.clone(),ids_pos,n_pos,strength_add,target_ids,apply_to_all_weights)
                    all_weights_neg = self.pos_neg_weights(all_weights.clone(),ids_neg,n_neg,strength_subtract,target_ids,apply_to_all_weights)
                    all_weights = all_weights + ((all_weights_pos - all_weights) - (all_weights_neg - all_weights)) * both_strengths_multiplier

                cond_stage_model.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(all_weights.to(device=device,dtype=cdtype))
        return clip,

class levelCLIPWeights:
    def __init__(self):
        self.clips_storage = {}
        msg = "USE A NEW NODE OR RESTART THE UI TO EDIT ANOTHER CLIP MODEL\n\
To reset the current model to its normal state turn the toggle off and run a batch before unplugging!\n\
Do not use with another similar node."
        print(f" \033[91m\n{msg}\033[0m")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip": ("CLIP",),
                "level_clip" : ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, level_clip):
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

                if level_clip:
                    for g in ignored_token_ids:
                        ignored_token_weights.append(all_weights[g].clone())
                    norm_w = torch.linalg.norm(all_weights,dim=1).unsqueeze(1)
                    backup_nan  = all_weights.clone()
                    all_weights = all_weights * (norm_w.mean() + 1e-16) / (norm_w + 1e-16)
                    nan_values  = torch.isnan(all_weights)
                    all_weights[nan_values] = backup_nan[nan_values]
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

# class saveCustomTokenNode:
#     def __init__(self):
#         pass
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#                 "clip": ("CLIP",),
#                 "target_text": ("STRING", {"multiline": True,"tooltip":"Which tokens are to be saved. Commas are ignored."}),
#                 "filename": ("STRING", {"multiline": False, "default":"filename"}),
#                 "save_ignored_tokens" : ("BOOLEAN", {"default": False}),
#             }
#         }

#     FUNCTION = "exec"
#     CATEGORY = "advanced/CLIP Weights Surgery"
#     RETURN_TYPES = ()
#     OUTPUT_NODE = True

#     def exec(self, clip, target_text, filename, save_ignored_tokens=False):
#         to_save = {}
#         for c, k in enumerate(["g","l"]):
#             csm1 = getattr(clip.cond_stage_model, f"clip_{k}", None)
#             tokenizer = getattr(clip.tokenizer, f"clip_{k}", None)
#             if csm1 is not None:
#                 to_save[k] = {}
#                 if not save_ignored_tokens:
#                     tids = words_to_tokens_ids(tokenizer, target_text)
#                 else:
#                     tids = ignored_token_ids
#                 weights = csm1.transformer.text_model.embeddings.token_embedding.weight
#                 for t in tids:
#                     to_save[k][t] = weights[t].clone().cpu()
#         if len(to_save) > 0:
#             file_path = os.path.join(weights_output_path, f"{filename}.pt")
#             torch.save(to_save,file_path)
#             print(f"Weights saved at path: {file_path}")
#         else:
#             print("The only compatible models are SD1.x/SD2/SDXL. Nothing will be saved.")
#         return {}

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
            required["loaded_weights_strength"] = ("FLOAT", {"default": 1, "min": -10.0, "max": 100.0, "step": 0.1})
        return {"required": required}

    IS_WEIGHTED_AVERAGE = False
    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, filename, output_original=True, loaded_weights_strength=0):
        clip = clip.clone()
        file_path = os.path.join(weights_output_path, f"{filename}.pt")
        custom_weights = torch.load(file_path)
        for k in custom_weights:
            csm1 = getattr(clip.cond_stage_model, f"clip_{k}", None)
            if csm1 is not None:
                clip_weights = csm1.transformer.text_model.embeddings.token_embedding.weight.clone()
                device = clip_weights.device
                for t in custom_weights[k]:
                    if k not in self.original_weights:
                        self.original_weights[k] = {}
                    if t in self.original_weights[k]:
                        if output_original:
                            clip_weights[t] = self.original_weights[k][t].clone().to(device=device)
                    else:
                        self.original_weights[k][t] = clip_weights[t].clone().to(device='cpu')
                    if loaded_weights_strength != 0:
                        old_weight = self.original_weights[k][t].clone().to(device=device)
                        new_weight = custom_weights[k][t].clone().to(device=device)
                        # new_weight = old_weight.norm() * new_weight / new_weight.norm()
                        clip_weights[t] = new_weight * loaded_weights_strength + old_weight * (1 - loaded_weights_strength)
                        # max(0, 1 - loaded_weights_strength)
                    if not output_original:
                        clip_weights[t] = custom_weights[k][t].clone().to(device=device)
                csm1.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(clip_weights)
        return clip,

class multiplyTokenNode:
    def __init__(self):
        self.clips_storage = {}
        msg = "USE A NEW NODE OR RESTART THE UI TO EDIT ANOTHER CLIP MODEL\n\
To reset the current model to its normal state turn the toggle off and run a batch before unplugging!\n\
Do not use with another similar node."
        print(f" \033[91m\n{msg}\033[0m")

    @classmethod
    def INPUT_TYPES(s):
        required  = {"clip": ("CLIP",),}
        required["target_text"] = ("STRING", {"multiline": True})
        required["multiplier"]  = ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step": 0.5})
        required["every_tokens"] = ("BOOLEAN", {"default": False})
        required["special_tokens_ids"] = ("STRING", {"multiline": False,"default":"49406, 49407, 0"})
        required["include_special"] = (["exclude","include","do_them_only"],)
        return {"required": required}

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip, target_text, multiplier, every_tokens, special_tokens_ids, include_special):
        clip = clip.clone()
        ignored_token_ids = [int(s) for s in special_tokens_ids.split(",") if s != ""]
        for c, k in enumerate(["g","l"]):
            cond_stage_model = getattr(clip.cond_stage_model, f"clip_{k}", None)
            if cond_stage_model is not None:
                clip_tokenizer = getattr(clip.tokenizer, f"clip_{k}", None)
                target_ids = words_to_tokens_ids(clip_tokenizer,target_text)
                weights = cond_stage_model.transformer.text_model.embeddings.token_embedding.weight
                device = weights.device
                cdtype = weights.dtype
                if k not in self.clips_storage:
                    self.clips_storage[k] = torch.clone(weights).to(device="cpu")
                    all_weights = torch.clone(weights).to(device=model_management.get_torch_device())
                else:
                    all_weights = torch.clone(self.clips_storage[k]).to(device=model_management.get_torch_device())
                special_tokens = []
                for s in ignored_token_ids:
                    special_tokens.append(all_weights[s].clone())
                if multiplier != 1:
                    if every_tokens and include_special != "do_them_only":
                        all_weights *= multiplier
                        if not include_special == "exclude":
                            for sid, s in enumerate(ignored_token_ids):
                                all_weights[s] = special_tokens[sid]
                    elif include_special == "do_them_only":
                        for t in ignored_token_ids:
                            all_weights[t] *= multiplier
                    else:
                        for t in target_ids:
                            all_weights[t] *= multiplier
                cond_stage_model.transformer.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(all_weights.to(device=device,dtype=cdtype))
        return clip,

NODE_CLASS_MAPPINGS = {
    "CLIP token injection": ClipTokenLobotomyNode,
    "CLIP token injection (alt version)": ClipTokenLobotomyAlternativeVersionNode,
    "CLIP take weights from another": replaceCLIPWeights,
    "CLIP level weights": levelCLIPWeights,
    "CLIP multiply weights": multiplyTokenNode,
    "CLIP fix nans (can be chained, best at end)": fixCLIPWeights,
    "CLIP save custom tokens": saveCustomTokenNode,
    "CLIP load custom tokens": type("loadCustomTokenNode", (loadCustomTokenNode,), { "IS_WEIGHTED_AVERAGE": False}),
    "CLIP load custom tokens (weighted average)": type("loadCustomTokenNodeWAvg", (loadCustomTokenNode,), { "IS_WEIGHTED_AVERAGE": True}),
}
