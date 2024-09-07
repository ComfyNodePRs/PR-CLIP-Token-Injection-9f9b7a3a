## Note:

These nodes are to edit the text weights, so to customize how the prompts will be interpreted.

The shared example weights does not contain any "knowledge" but the vector representations of the words affected.

I do not provide any guarantee of result regarding AI safety. This is all experimental.

You will find them in the category "advanced/token surgery".

# CLIP-Token-Injection

![image](https://github.com/user-attachments/assets/b4adb747-9cf4-4b56-b7c2-a97acc7fb0c4)

Each token composing any word in the first text input will be modified by those in the second and third text inputs.

"pos_vs_neg" affects the relative influence in between the second and third text inputs. At 1 will only add the meaning of the second text input. At 0 will only subtract what is the third text input.

"strength": This is a weighted average. At 1 will fully replace the target tokens. At 0 will have no effect.

The node can be chained with others and the results can be saved.
