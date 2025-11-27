import os
import sys

# Files extensions to clean up
CLEAN_EXTS = [
    '.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', 
    '.log', '.out', '.xdv', '.synctex.gz', '.toc'
]

# Size limit for GitHub warning (50MB in bytes)
LARGE_FILE_LIMIT = 50 * 1024 * 1024

def clean_artifacts():
    print("🧹 Cleaning up LaTeX build artifacts...")
    count = 0
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
            
        for file in files:
            if any(file.endswith(ext) for ext in CLEAN_EXTS):
                path = os.path.join(root, file)
                try:
                    os.remove(path)
                    print(f"   Deleted: {path}")
                    count += 1
                except OSError as e:
                    print(f"   Error deleting {path}: {e}")
    
    if count == 0:
        print("   No artifacts found to clean.")
    else:
        print(f"   Done. Removed {count} files.")

def check_large_files():
    print("\n⚖️  Checking for large files (>50MB)...")
    found_large = False
    for root, dirs, files in os.walk('.'):
        if '.git' in dirs:
            dirs.remove('.git')
            
        for file in files:
            path = os.path.join(root, file)
            try:
                size = os.path.getsize(path)
                if size > LARGE_FILE_LIMIT:
                    print(f"   ⚠️  WARNING: {path} is {size / (1024*1024):.2f} MB")
                    found_large = True
            except OSError:
                pass
                
    if not found_large:
        print("   All files look good for GitHub!")
    else:
        print("   Please review the warnings above before pushing.")

def check_pdf():
    print("\n📄 Checking for main.pdf...")
    if os.path.exists('main.pdf'):
        print("   ✅ main.pdf exists.")
    else:
        print("   ❌ main.pdf NOT found. Did you compile the paper?")

if __name__ == "__main__":
    print("🚀 Preparing repository for release...\n")
    clean_artifacts()
    check_large_files()
    check_pdf()
    print("\n✨ Done!")
