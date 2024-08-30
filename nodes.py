import torch
import comfy.model_management as model_management

mean = lambda *x: sum(x)/len(x)
ignored_token_ids = [49406, 49407, 0]
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
                # "apply_to_all_weights" : ("BOOLEAN", {"default": False, "tooltip":"This is a bad idea."}),
            }
        }

    FUNCTION = "exec"
    CATEGORY = "advanced/CLIP Weights Surgery (do not chain)"
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
             strength_sub, both_strengths_multiplier, allow_doubles):
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
                        # max(1, p_div * strength_add - n_div * strength_sub) / 4
                        if t in weight_difs:
                            weight_difs[t] += (p_res - n_res) * both_strengths_multiplier
                        else:
                            weight_difs[t]  = (p_res - n_res) * both_strengths_multiplier
                    for t in target_ids:
                        all_weights[t] = all_weights[t] + weight_difs[t]
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
    CATEGORY = "advanced/CLIP Weights Surgery (do not chain)"
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
                    all_weights = all_weights * torch.nan_to_num(norm_w.mean() / norm_w, nan=1.0)
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
    CATEGORY = "advanced/CLIP Weights Surgery (do not chain)"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip_to_modify, clip_to_pick_from, target_text, output_original):
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
    CATEGORY = "advanced/CLIP Weights Surgery (do not chain)"
    RETURN_TYPES = ("CLIP",)

    def exec(self, clip_to_fix, clip_doctor):
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

NODE_CLASS_MAPPINGS = {
    "CLIP token injection": ClipTokenLobotomyNode,
    "CLIP take weights from another": replaceCLIPWeights,
    "CLIP level weights": levelCLIPWeights,
    "CLIP fix nans (can be chained, best at end)": fixCLIPWeights,
}
