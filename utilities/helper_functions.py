## General helper functions
import re
import logging
from accelerate.logging import get_logger
import diffusers
from diffusers import StableDiffusionPipeline

logger = get_logger(__name__, log_level="INFO")

def text_to_prompt(text:str)->str:
    text = re.sub('[^\w\s]',' ',text)
    # prompt = f'''The book is about {text}.The cover of the book is attractive and shows stunning details,
    #            photorealistic,rectangular aspect ratio,Cinematic and volumetric ligh,margins on both side'''

    prompt = f'''The book is about: {text}. Its cover captivates with intricate details, boasting photorealism and a 
    rectangular aspect ratio. Enhanced with cinematic and volumetric lighting, it features balanced margins on both sides'''

    #prompt = f'''Centered on {text}, the book's cover boasts vivid imagery, a rectangular layout, and evenly spaced margins, contributing to its overall visual appeal.'''

    return prompt

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


#
def string_to_bool(input_string):
    normalized_string = input_string.strip().lower()
    
    true_values = ["true", "True", "TRUE" , "T" "yes", "1", "t", "y"]
    
    false_values = ["false", "False" , "FALSE", "F", "0", "f", "n"]
    
    if normalized_string in true_values:
        return True
    elif normalized_string in false_values:
        return False
    else:
        raise ValueError("Input string does not represent a boolean value")



