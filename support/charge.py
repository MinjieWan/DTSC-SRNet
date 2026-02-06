from PIL import Image
import os


def expand_image(input_path, output_path=None):
    """
    Expand 384x384 image to 3072x3072 through pixel repetition
    """
    # Open image
    img = Image.open(input_path)

    # Ensure RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Calculate scaling factor (3072 / 384 = 8)
    scale = 8

    # Use nearest neighbor interpolation (pixel repetition) for expansion
    new_size = (img.width * scale, img.height * scale)
    expanded_img = img.resize(new_size, Image.NEAREST)

    # Save image
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_3072x3072.png"

    expanded_img.save(output_path)
    print(f"Saved: {output_path} ({expanded_img.size})")
    return expanded_img


# Usage example
if __name__ == "__main__":
    # Process single image
    expand_image("1191_SR.bmp", "output1191MISR.png")

    # Or batch processing (example)
    # for img_file in os.listdir("images_folder"):
    #     if img_file.endswith((".png", ".bmp")):
    #         expand_image(f"images_folder/{img_file}")