import zipfile, glob, os

# Extract every zip (including nested .csv.zip) into this folder, then remove the zips.
while glob.glob("*.zip"):
    for z in glob.glob("*.zip"):
        print("extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(".")
        os.remove(z)

print("done ->", [f for f in os.listdir(".") if f.endswith(".csv")])