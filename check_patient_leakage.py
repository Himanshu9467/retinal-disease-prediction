from pathlib import Path
import re

def get_patient_id(filename):
    name = Path(filename).stem.lower()

    match = re.search(r'processed_dataset_(\d+)', name)
    if match:
        return match.group(1)

    return name

def collect_patients(split_dir):
    patients = set()

    for img in Path(split_dir).rglob("*.*"):
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            patients.add(get_patient_id(img.name))

    return patients

train_patients = collect_patients("research_binary_7000/train")
val_patients = collect_patients("research_binary_7000/val")
test_patients = collect_patients("research_binary_7000/test")

print("Train patients:", len(train_patients))
print("Val patients:", len(val_patients))
print("Test patients:", len(test_patients))

print("\nTrain ∩ Val =", len(train_patients.intersection(val_patients)))
print("Train ∩ Test =", len(train_patients.intersection(test_patients)))
print("Val ∩ Test =", len(val_patients.intersection(test_patients)))

if (
    len(train_patients.intersection(val_patients)) == 0
    and len(train_patients.intersection(test_patients)) == 0
    and len(val_patients.intersection(test_patients)) == 0
):
    print("\n✅ NO PATIENT LEAKAGE DETECTED")
else:
    print("\n❌ PATIENT LEAKAGE DETECTED")