### From https://towardsdatascience.com/florence-2-mastering-multiple-vision-tasks-with-a-single-vlm-model-435d251976d0

#Load model:
model_id = ‘microsoft/Florence-2-large’
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

#Load image:
image = Image.open(img_path)

def run_example(image, task_prompt, text_input=''):

    prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors=”pt”).to(‘cuda’, torch.float16)

    generated_ids = model.generate(
        input_ids=inputs[“input_ids”].cuda(),
        pixel_values=inputs[“pixel_values”].cuda(),
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
        early_stopping=False,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

print (run_example(image, task_prompt='<CAPTION>'))
# Output: 'A black camera sitting on top of a wooden table.'

print (run_example(image, task_prompt='<DETAILED_CAPTION>'))
# Output: 'The image shows a black Kodak V35 35mm film camera sitting on top of a wooden table with a blurred background.'

print (run_example(image, task_prompt='<MORE_DETAILED_CAPTION>'))
# Output: 'The image is a close-up of a Kodak VR35 digital camera. The camera is black in color and has the Kodak logo on the top left corner. The body of the camera is made of wood and has a textured grip for easy handling. The lens is in the center of the body and is surrounded by a gold-colored ring. On the top right corner, there is a small LCD screen and a flash. The background is blurred, but it appears to be a wooded area with trees and greenery.'
