## Note:

These nodes are to edit the text weights, so to customize how the prompts will be interpreted.

The shared example weights does not contain any image-knowledge but the text vector of the words affected.

> [!CAUTION]
> I do not provide any guarantee of result regarding AI safety. This is all experimental.

You will find them in the category "advanced/token surgery".

# CLIP-Token-Injection

![image](https://github.com/user-attachments/assets/b4adb747-9cf4-4b56-b7c2-a97acc7fb0c4)

Each token composing any word in the first text input will be modified by those in the second and third text inputs.

"pos_vs_neg" affects the relative influence in between the second and third text inputs. At 1 will only add the meaning of the second text input. At 0 will only subtract what is the third text input. This value has no use if only one of the two text inputs is used.

"strength": This is a weighted average. At 1 will fully replace the target tokens. At 0 will have no effect.

The node can be chained with others and the results can be saved.

# Save / Load custom tokens:

![image](https://github.com/user-attachments/assets/278933ab-4008-4250-a605-936a394d81a6)

Allows to save directly the customized weights by simply typing them.

The node loading them uses a weighted average. 1 meaning full replacement of the target tokens.

## Potential use case

### Weights customization:

Allows to edit how the text will be interpreted. In cases where you would be prefering something differently.

Temporarily or permanently.

You could for example combine into realistic "dark noise shadow cinematic photographic" and subtract from it "cgi render rendering rendered videogame videogames cartoon" to help with general realism.

### AI Safety:

> [!CAUTION]
> **I DO NOT PROVIDE ANY GUARANTEE OF RESULT** nor that any user won't be able to circumvent any modification you may apply.

Provided in the "custom weights" folder a large set of tokens where main concepts relative to nudity/pornography and children have been modified.

Any user prompting for unsafe content will find Chris Hansen staring at him, with some FBI-related details.

The file named "naked_little_girl_example.png" shows you how to load these weights.

I can not share the full list of words used as even pastebin thinks that I'm trying to share CP.

The child related targeted tokens from the file named "AISafety.pt" are the following:

    girl teen teens teenager boy toddlers children infant infants baby babies kid kiddo yo years old
    1 2 3 4 5 6 7 8 9 0
    one two three four five six seven eight nine ten eleven twelve

Some have been replaced by "Chris" and some by "Hansen". Tokens used by NSFW related terms have mainly been replaced by the token "fbi".
