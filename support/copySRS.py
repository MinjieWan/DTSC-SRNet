import os
import shutil


def copy_first_image_from_subfolders(source_dir, target_dir):
    """
    Copy lr_frame_00.png from all subfolders in source directory to target directory,
    using subfolder names as new filenames (keeping .png extension)

    Parameters:
    source_dir: Source root folder path (contains all subfolders)
    target_dir: Target folder path (for storing copied images)
    """
    # 1. Create target folder if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # 2. Iterate through all items in source directory
    for subfolder_name in os.listdir(source_dir):
        # Construct full path of subfolder
        subfolder_path = os.path.join(source_dir, subfolder_name)

        # Skip non-folder items (e.g., files directly in source root)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping non-folder item: {subfolder_path}")
            continue

        # 3. Construct source image path (lr_frame_00.png in subfolder)
        src_image_path = os.path.join(subfolder_path, "lr_frame_05.bmp")

        # Check if image exists
        if not os.path.exists(src_image_path):
            print(f"Warning: lr_frame_00.png not found in {subfolder_path}, skipping folder")
            continue

        # 4. Construct target image path (target folder + subfolder name + .png)
        target_image_name = f"{subfolder_name}.bmp"
        target_image_path = os.path.join(target_dir, target_image_name)

        # 5. Copy image (copy2 preserves file metadata, more friendly than copy)
        try:
            shutil.copy2(src_image_path, target_image_path)
            print(f"Successfully copied: {src_image_path} -> {target_image_path}")
        except Exception as e:
            print(f"Failed to copy: {src_image_path}, Error: {str(e)}")


# ===================== Main Program Entry =====================
if __name__ == "__main__":
    # Replace with actual source and target folder paths
    # Windows path example: r"D:\source_folder" (r avoids escape sequences)
    # Linux/Mac path example: "/home/user/source_folder"
    SOURCE_FOLDER = "lr_output_testv8+"
    TARGET_FOLDER = "lr_output_testv8+1t5"

    # Execute copy function
    copy_first_image_from_subfolders(SOURCE_FOLDER, TARGET_FOLDER)
    print("\nCopy task completed!")