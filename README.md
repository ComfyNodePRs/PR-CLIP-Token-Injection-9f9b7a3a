Update:

To port all the "score" tags in a CLIP as a single token meant to be used as a negative set the nodes:

~~CLIP Load ==> my node in here named like ~~"load custom weight (perp)" using the file named "all_scores_as_one_negative.pt" and a strength of 0.5, nothing more nothing less~~ Just use the load token, the simple thing. ==> save clip ==> restart the  UI because memory shenanigans (I did it kinda hacky) ==> enjoy~~
 Just load the workflow

The secret single negative token is written in the text file in the weights folder.

The category in is "advanced" and something like "CLIP surgery"

# CLIP-Token-Injection
Manually edit CLIP token weights, results can be saved to remain permanent.

Work in progress but functional!

update: save/load weights. Added "scores" tags in the custom weights folder.

Don't chain these nodes. Save the result after testing, restart and reload. Don't load a different model into the same node neither.

(they keep the weights they receive first)

## Note

Current state is hacky as hell. Don't change model, limit to one node and everything will be alright.


## CLIP token injection:

![image](https://github.com/user-attachments/assets/e2dc0f4e-3490-402d-b982-0aa4b45788bf)

Here the word "woman" is targeted. The token weight corresponding to it will become:

"woman" + ((("pretty beautiful happy" - "woman") * positive slider) - (("evil creepy old" - "woman") * negative slider)) * both_strengths_multiplier

The effect will become permanent if the clip model is saved. Every "woman" will become pretty beautiful and happy. What else do you want?

So far compatible with sd1.x, sd2 and SDXL.

Base image / Happy and stuff / In the third one I switched the positive with the negative text:

![05190UI_00001_](https://github.com/user-attachments/assets/63f8b390-d024-4cfe-8f8a-7fb7efc9266d)

![05191UI_00001_](https://github.com/user-attachments/assets/052c7415-b9d9-422e-b096-c797e78c7e84)

![05192UI_00001_](https://github.com/user-attachments/assets/8c45bc6b-255b-4161-9f11-620a5a17ee79)

Multiple words can be used in the first text box to target multiple tokens at once.

If the same tokens are used in the target and in the positive or negative inputs the will be weighted against those which are differents.

For example, you may write the same words in the target and positive, and their meaning will be more diluted in between each other.

The opposite is also true. I tested so far to bias tokens with many and the results can be great.

## CLIP take weights from another:

![image](https://github.com/user-attachments/assets/dbe68f79-dd69-41e8-9fae-5e98216da8b3)

It's self-explained!

The model to pick from can be changed. Not the one being modified. So you can test the results obtained with different models until you wish to save your result. This can be useful to take the weights from another one if there were trained words maybe that you would like to apply to your favorite merge.

## CLIP level weights:

![image](https://github.com/user-attachments/assets/df86ed00-28fe-4d8b-8629-b9e4c38e2d4d)

Divide each token by it's own torch.norm() and multiply by the mean. After a bit of testing this really seems to help my most tortured models to stop merging people with furnitures but my confidence is relative. I've only been testing all of this since a few hours. Test and see.

The toggles allows to compare.

## CLIP fix nans:

![image](https://github.com/user-attachments/assets/2a9235c5-c950-4577-b1a6-1f0c85226472)


This one can be chained, does not keep anything in memory and simply replace any potential NaN from the first model by those in the second input.

I made it as I was testing the concepts above but have not had to use it since I made it 🙄
