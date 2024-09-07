## Note:

These nodes are to edit the text vectors of [CLIP](https://github.com/openai/CLIP) models, so to customize how the prompts will be interpreted.

You could see it as either customisation, "one token prompt" up to some limitation and a way to mess with how the text will be interpreted.

The edited CLIP can then be saved, or as well the edited tokens themselves.

The shared example weights does not contain any image-knowledge but the text vector of the words affected.

You will find them in the category "advanced/token surgery".

Here the words "woman" and "forest" have been affected. As you can see you can shove a lot of meaning in single tokens:

![image_grid](https://github.com/user-attachments/assets/0b06a27a-8d09-4497-a023-f8414632dbe2)

This example is explained further below.


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

You could for example combine into "realistic" the tokens "dark noise shadow cinematic photographic" and subtract from it "cgi render rendering rendered videogame videogames cartoon" to help with general realism.

### Example

Prompt:

    a woman with long hair in a forest, photography

Here "woman" is being added "pretty perfect love" and subtracted "evil demon ugly" and "forest" is being modified the same way:

![image](https://github.com/user-attachments/assets/45ea18e4-9290-4dea-af7e-6c8f53e2158e)

Same modification strength for both words, In this order: -1 / -0.7 / 0 (base image) / 0.7 / 1 / 1 but fully adding the words to add, without subtracting the negative words (so "pos_vs_neg" at 1):

![image_grid](https://github.com/user-attachments/assets/ae81eed4-cc99-492c-a66c-e17fa45b9763)

### AI Safety:

> [!CAUTION]
> **I DO NOT PROVIDE ANY GUARANTEE OF RESULT** nor that any user won't be able to circumvent any modification you may apply.

Provided in the "custom weights" folder a file named "AISafety.pt" in which a large set of tokens where main concepts relative to nudity/pornography and children have been modified.

Any user prompting for unsafe content will find Chris Hansen staring at him, with some FBI-related details.

The file named "ai_safety_example.png" shows you how to load these weights and demonstrates what you will get if you try to use such prompt.

I can not share the full list of words used as even pastebin thinks that I'm trying to share CP.

The child related targeted tokens from the file named "AISafety.pt" are the following:

    girl teen teens teenager boy toddlers children infant infants baby babies kid kiddo yo years old
    1 2 3 4 5 6 7 8 9 0
    one two three four five six seven eight nine ten eleven twelve

Some have been replaced by "Chris" and some by "Hansen". Tokens used by NSFW related terms have mainly been replaced by the token "fbi".

This is what you will obtain if you attempt to prompt for unsafe content with such modification:

![01383UI_00001_](https://github.com/user-attachments/assets/6d60449a-1d50-42ae-9401-6cb25531c6ef)


> [!TIP]
> - Remember, if you are attempting to use many tokens to add and subtract that it will throw it all into single token. Typing the trylogy of the lord of the rings will most likely give you a token without much meaning.
> - in the add or subtract text inputs, a word used twice will be counting double
> - to not fully lose the meaning of a word you can either set a lower strength or copy the target token in the text input of tokens to add.
> - Using words that are a single token seems to work better. CLIP L and CLIP G shares the same vocabulary which you can check in any Huggingface repository containing a CLIP L or G model, often under the name of "vocab.json". [Here is one](https://huggingface.co/openai/clip-vit-large-patch14/raw/main/vocab.json)
