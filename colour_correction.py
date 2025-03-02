from PIL import Image, ImageEnhance

input_path = "water_fixed.jpg"
img = Image.open(input_path)

#reduce brightness
brightness_enhancer = ImageEnhance.Brightness(img)
darkened_img = brightness_enhancer.enhance(0.7)

#lower saturation
color_enhancer = ImageEnhance.Color(darkened_img)
desaturated_img = color_enhancer.enhance(0.5)

#increase contrast
contrast_enhancer = ImageEnhance.Contrast(desaturated_img)
contrast_img = contrast_enhancer.enhance(1.5)
output_path = "output.jpg"
contrast_img.save(output_path)
print("SAVED")
