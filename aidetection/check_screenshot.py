import os

data_dir = "dataset/"
categories = ["text_field", "email_field", "dropdown", "checkbox", "radio_button", "button"]

for category in categories:
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        print(f"⚠️ Missing folder: {path}")
        continue

    num_images = len(os.listdir(path))
    print(f"📂 {category}: {num_images} images")
