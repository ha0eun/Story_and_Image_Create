#https://github.com/aniketmaurya/stable_diffusion_inference.git

from stable_diffusion_inference import create_text2image
from translate import Translator


text2image = create_text2image("sd1")
# text2image = create_text2image("sd2_high")  # for SD 2.0 with 768 image size
# text2image = create_text2image("sd2_base")  # for SD 2.0 with 512 image size


translator = Translator(from_lang="ko", to_lang="en")
translation = translator.translate("A cartoon of 호랑이와 곰이 서로 싸우고 다투고 있는데")
print(translation)

image = text2image(translation, image_size=256, inference_steps=30)
image.save("TRANS.png")
