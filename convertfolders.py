import os
import shutil

# === CONFIGURATION ===
source_root = r"C:\Users\Janvi\Desktop\APPROACH1\dataset"  # <-- Your current dataset root
target_root = r"C:\Users\Janvi\Desktop\APPROACH1\relativesdataset"          # <-- Where you want the new structure

# Create target root folder if it doesn't exist
os.makedirs(target_root, exist_ok=True)

# Loop through each patient folder
for patient in os.listdir(source_root):
    patient_path = os.path.join(source_root, patient)

    if os.path.isdir(patient_path):
        for relative in os.listdir(patient_path):
            relative_path = os.path.join(patient_path, relative)

            if os.path.isdir(relative_path):
                # Construct new folder name: PatientX_RelativeY
                new_folder_name = f"{patient}_{relative}"
                new_folder_path = os.path.join(target_root, new_folder_name)

                # Copy all images to the new folder
                os.makedirs(new_folder_path, exist_ok=True)
                for file in os.listdir(relative_path):
                    src_file = os.path.join(relative_path, file)
                    dst_file = os.path.join(new_folder_path, file)
                    shutil.copy2(src_file, dst_file)

                print(f"✅ Copied {patient}/{relative} to {new_folder_name}")

print("\n🎉 Dataset restructuring complete!")
