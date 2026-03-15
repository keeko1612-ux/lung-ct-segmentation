# 🐍 Python Tutorial: Understanding the Lung CT Segmentation Code

Welcome! This tutorial will teach you both **Python programming fundamentals** and how this **medical imaging code works**. By the end, you'll understand every line of code in this project.

---

## 📚 Table of Contents

1. [Python Basics](#1-python-basics)
2. [Understanding the Imports](#2-understanding-the-imports)
3. [Loading DICOM Medical Images](#3-loading-dicom-medical-images)
4. [Converting to Hounsfield Units](#4-converting-to-hounsfield-units)
5. [Lung Segmentation Algorithm](#5-lung-segmentation-algorithm)
6. [Nodule Detection](#6-nodule-detection)
7. [Visualization with Matplotlib](#7-visualization-with-matplotlib)
8. [Complete Code Reference](#8-complete-code-reference)

---

## 1. Python Basics

Before diving into the code, let's learn the Python concepts used in this project.

### 1.1 Variables and Data Types

```python
# Variables store data - no need to declare type
patient_name = "John"          # String (text)
slice_count = 324              # Integer (whole number)
pixel_spacing = 0.7            # Float (decimal number)
is_processed = True            # Boolean (True/False)

# Print values to see them
print(f"Patient: {patient_name}")
print(f"Slices: {slice_count}")
```

**What's `f"..."`?** This is an **f-string** (formatted string). It lets you embed variables directly:
```python
name = "Alice"
age = 25
print(f"Hello, {name}! You are {age} years old.")
# Output: Hello, Alice! You are 25 years old.
```

### 1.2 Lists

Lists store multiple items in order:

```python
# Create a list
slices = [10, 20, 30, 40, 50]

# Access items (counting starts at 0!)
first_slice = slices[0]    # → 10
third_slice = slices[2]    # → 30
last_slice = slices[-1]    # → 50 (negative counts from end)

# Add items
slices.append(60)          # Now: [10, 20, 30, 40, 50, 60]

# List length
print(len(slices))         # → 6
```

### 1.3 For Loops

Loops let you repeat actions:

```python
# Loop through a list
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(num * 2)
# Output: 2, 4, 6, 8, 10

# Loop with index using enumerate
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"Item {index}: {fruit}")
# Output:
# Item 0: apple
# Item 1: banana
# Item 2: cherry
```

### 1.4 Functions

Functions are reusable blocks of code:

```python
def greet(name):
    """This is a docstring - it describes the function."""
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
print(message)  # → Hello, Alice!

# Function with multiple parameters
def add_numbers(a, b):
    result = a + b
    return result

total = add_numbers(5, 3)  # → 8
```

### 1.5 Comments

```python
# This is a single-line comment

"""
This is a multi-line comment (docstring).
It can span multiple lines.
Often used to describe functions.
"""

x = 10  # You can also add comments at end of line
```

### 1.6 Comparison and Logic

```python
# Comparison operators
x = 10
print(x > 5)    # True  (greater than)
print(x < 5)    # False (less than)
print(x == 10)  # True  (equal to)
print(x != 5)   # True  (not equal to)
print(x >= 10)  # True  (greater or equal)
print(x <= 10)  # True  (less or equal)

# Logical operators
a = True
b = False
print(a and b)  # False (both must be True)
print(a or b)   # True  (at least one True)
print(not a)    # False (opposite)

# Used in conditions
if x > 5 and x < 15:
    print("x is between 5 and 15")
```

### 1.7 Dictionaries

Dictionaries store key-value pairs:

```python
patient = {
    "name": "John Doe",
    "age": 45,
    "scan_date": "2024-01-15"
}

# Access values by key
print(patient["name"])     # → John Doe
print(patient["age"])      # → 45

# Add new key-value
patient["diagnosis"] = "Normal"
```

---

## 2. Understanding the Imports

The first cell imports all necessary libraries:

```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure, morphology
import scipy.ndimage as ndimage
```

Let's break down each one:

### `import pydicom`
**DICOM** = Digital Imaging and Communications in Medicine  
This is the standard format for medical images (CT, MRI, X-ray).  
`pydicom` reads these `.dcm` files and extracts:
- The actual image data (pixel array)
- Metadata (patient info, scan parameters)

### `import numpy as np`
**NumPy** = Numerical Python  
The foundation for scientific computing in Python.

```python
import numpy as np

# Create arrays (like super-powered lists)
arr = np.array([1, 2, 3, 4, 5])

# Math operations work on ALL elements at once
print(arr * 2)      # → [2, 4, 6, 8, 10]
print(arr + 10)     # → [11, 12, 13, 14, 15]
print(arr.mean())   # → 3.0 (average)
print(arr.min())    # → 1
print(arr.max())    # → 5

# 2D arrays (images!)
image = np.array([
    [0, 100, 200],
    [50, 150, 250],
    [25, 125, 225]
])
print(image.shape)  # → (3, 3) - 3 rows, 3 columns
```

**Why `as np`?** It's an alias - shorter to type `np.array()` vs `numpy.array()`.

### `import matplotlib.pyplot as plt`
**Matplotlib** creates visualizations (graphs, images).

```python
import matplotlib.pyplot as plt

# Display an image
plt.imshow(my_image, cmap='gray')
plt.title('My Image')
plt.show()

# Create multiple plots
fig, axes = plt.subplots(1, 3)  # 1 row, 3 columns
axes[0].imshow(image1)
axes[1].imshow(image2)
axes[2].imshow(image3)
plt.show()
```

### `from pathlib import Path`
**Path** handles file/folder paths cleanly:

```python
from pathlib import Path

folder = Path("data/patient_001")

# Find all DICOM files
for file in folder.glob("*.dcm"):
    print(file)
# Output: data/patient_001/slice001.dcm
#         data/patient_001/slice002.dcm
#         etc.
```

### `from skimage import measure, morphology`
**scikit-image** provides image processing functions:
- `measure.label()` - identifies connected regions
- `measure.regionprops()` - measures properties of regions
- `morphology.remove_small_objects()` - removes noise

### `import scipy.ndimage as ndimage`
**SciPy** provides scientific functions:
- `ndimage.binary_fill_holes()` - fills holes in binary masks

---

## 3. Loading DICOM Medical Images

```python
# Step 1: Define the folder path
DICOM_FOLDER = "data/patient_001/CT-SCAN-FOLDER"

# Step 2: Create empty list to store slices
slices = []

# Step 3: Loop through all .dcm files in the folder
for f in sorted(Path(DICOM_FOLDER).glob("*.dcm")):
    ds = pydicom.dcmread(str(f))  # Read the DICOM file
    slices.append(ds)              # Add to our list

# Step 4: Sort slices by position (z-axis)
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# Step 5: Print info about what we loaded
print(f"Loaded {len(slices)} slices")
print(f"Image size: {slices[0].pixel_array.shape}")
print(f"Slice thickness: {slices[0].SliceThickness} mm")
```

### Line-by-Line Explanation:

**`sorted(Path(DICOM_FOLDER).glob("*.dcm"))`**
- `Path(DICOM_FOLDER)` - creates a Path object
- `.glob("*.dcm")` - finds all files ending in `.dcm`
- `sorted(...)` - sorts files alphabetically

**`pydicom.dcmread(str(f))`**
- `str(f)` - converts Path to string
- `dcmread()` - reads the DICOM file
- Returns a Dataset object with:
  - `.pixel_array` - the actual image data
  - `.ImagePositionPatient` - 3D position [x, y, z]
  - `.SliceThickness` - how thick each slice is (mm)
  - Many more metadata fields!

**`slices.sort(key=lambda x: ...)`**

This is a **lambda function** - a small inline function:
```python
# These are equivalent:
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# Same as writing:
def get_z_position(x):
    return float(x.ImagePositionPatient[2])
slices.sort(key=get_z_position)
```

**Why sort by z-position?**  
CT scans are 3D volumes. Each slice has a z-coordinate. Sorting ensures slices are in correct anatomical order (bottom to top of body).

---

## 4. Converting to Hounsfield Units

### What are Hounsfield Units (HU)?

CT scanners measure how much X-rays are absorbed by tissue. These are stored as raw numbers, then converted to **Hounsfield Units**:

| Material | HU Value |
|----------|----------|
| Air | -1000 |
| Lung | -500 |
| Fat | -100 |
| Water | 0 |
| Soft tissue | +40 to +80 |
| Bone | +400 to +1000 |

### The Conversion Code

```python
def to_hounsfield(slices):
    # Stack all slice images into a 3D volume
    # astype(int32) is CRITICAL — prevents integer overflow!
    volume = np.stack([s.pixel_array.astype(np.int32) for s in slices])
    
    # Get conversion factors from DICOM metadata
    slope = float(slices[0].RescaleSlope)      # Usually 1
    intercept = float(slices[0].RescaleIntercept)  # Usually -1024
    
    # Apply the linear transformation
    return volume * slope + intercept

hu_volume = to_hounsfield(slices)
```

### Breaking It Down:

**`np.stack([s.pixel_array.astype(np.int32) for s in slices])`**

This is a **list comprehension** - a compact way to create lists:
```python
# This list comprehension:
[s.pixel_array.astype(np.int32) for s in slices]

# Is equivalent to:
result = []
for s in slices:
    pixel_array = s.pixel_array        # Get image array
    converted = pixel_array.astype(np.int32)  # Convert type
    result.append(converted)
```

**`.astype(np.int32)`** converts data type.

⚠️ **CRITICAL**: Raw DICOM data is `uint16` (unsigned integer, 0 to 65535).  
HU values can be **negative** (air = -1000).  
Without conversion to `int32`, negative values would overflow!

```python
# Example of the problem:
uint16_max = 65535
uint16_value = np.uint16(0)
result = uint16_value - 1024  # Should be -1024
# But uint16 can't store -1024, so it wraps around!
```

**`np.stack()`** combines 2D arrays into a 3D volume:
```python
# If we have 324 slices of 512×512:
# After stacking: shape = (324, 512, 512)
#                         ↑    ↑    ↑
#                    slices  rows  columns
```

### Lung Windowing

```python
def apply_lung_window(volume):
    low, high = -1350, 150
    return np.clip(volume, low, high)

windowed = apply_lung_window(hu_volume)
```

**`np.clip(volume, low, high)`**  
Limits values to a range:
- Values below -1350 become -1350
- Values above 150 become 150
- Values in between stay unchanged

This "window" focuses on lung tissue (-1000 to -500 HU), making it visible while ignoring very bright bones.

---

## 5. Lung Segmentation Algorithm

This is the core of the project! Let's understand how we automatically identify lungs in CT images.

```python
def segment_lungs(hu_volume, slice_idx):
    """
    Segment (isolate) the lung regions from a CT slice.
    
    Parameters:
        hu_volume: 3D array of Hounsfield Unit values
        slice_idx: which slice to process (0 to 323)
    
    Returns:
        img: the original CT image
        lungs_final: binary mask where True = lung
    """
    # Get one 2D slice from the 3D volume
    img = hu_volume[slice_idx]
    
    # === STEP 1: Create body mask ===
    # Everything brighter than -700 HU is inside the patient
    body_mask = img > -700
    
    # === STEP 2: Fill the body ===
    # The lungs are holes in the body - fill them temporarily
    body_filled = ndimage.binary_fill_holes(body_mask)
    
    # === STEP 3: Find lungs inside body ===
    # Lungs are dark (< -400 HU) AND inside the body
    lungs_inside = body_filled & (img < -400)
    
    # === STEP 4: Remove small noise ===
    # True lungs are large - remove small blobs
    lungs_clean = morphology.remove_small_objects(lungs_inside, min_size=2000)
    
    # === STEP 5: Fill holes in lungs ===
    # Blood vessels inside lungs create small holes - fill them
    lungs_final = ndimage.binary_fill_holes(lungs_clean)
    
    # === STEP 6: Measure the regions ===
    labels = measure.label(lungs_final)  # Give each connected region a number
    regions = measure.regionprops(labels)  # Measure each region
    regions.sort(key=lambda x: x.area, reverse=True)  # Sort by size
    
    return img, lungs_final
```

### Understanding Boolean Operations

```python
# Boolean arrays (True/False for each pixel)
body_mask = img > -700  # True where pixel > -700, False otherwise

# Logical AND (&) - BOTH conditions must be True
lungs_inside = body_filled & (img < -400)
```

**Visual example:**
```
body_filled:  [[T, T, T],    (img < -400): [[F, T, F],    Result (AND):  [[F, T, F],
               [T, T, T],                   [T, T, T],                    [T, T, T],
               [T, T, T]]                   [F, T, F]]                    [F, T, F]]
```

### Understanding `measure.label()`

This identifies **connected regions** - pixels that touch each other:

```python
# Input binary mask:
# 1 1 0 0 1 1
# 1 1 0 0 1 1
# 0 0 0 0 0 0
# 0 0 1 1 0 0

# After labeling:
# 1 1 0 0 2 2    ← Region 1 (left lung)
# 1 1 0 0 2 2    ← Region 2 (right lung)
# 0 0 0 0 0 0
# 0 0 3 3 0 0    ← Region 3 (maybe noise)
```

### Understanding `measure.regionprops()`

Returns measurements for each labeled region:

```python
regions = measure.regionprops(labels)

for r in regions:
    print(r.area)      # Number of pixels
    print(r.centroid)  # Center point (row, col)
    print(r.bbox)      # Bounding box
```

---

## 6. Nodule Detection

Nodules are small dense masses that might indicate cancer. This code finds them:

```python
def detect_nodule(hu_volume, slice_idx):
    """
    Detect potential lung nodules in a CT slice.
    
    Nodules are:
    - Inside the lung
    - Dense (brighter than normal lung tissue)
    - Small to medium sized
    """
    img = hu_volume[slice_idx]
    
    # First, segment the lungs (same as before)
    body_mask = img > -700
    body_filled = ndimage.binary_fill_holes(body_mask)
    lungs_inside = body_filled & (img < -400)
    lungs_clean = morphology.remove_small_objects(lungs_inside, min_size=2000)
    lung_mask = ndimage.binary_fill_holes(lungs_clean)
    
    # === Find dense regions INSIDE lungs ===
    # Nodules are denser than lung (-100 to 200 HU)
    potential_nodule = lung_mask & (img > -100) & (img < 200)
    
    # Remove tiny noise (< 10 pixels)
    potential_nodule = morphology.remove_small_objects(potential_nodule, min_size=10)
    
    # Remove large structures (blood vessels, > 2000 pixels)
    potential_nodule = morphology.remove_small_objects(~potential_nodule, min_size=2000)
    potential_nodule = ~potential_nodule  # Invert back
    
    # Label and measure candidates
    labels = measure.label(potential_nodule)
    regions = measure.regionprops(labels)
    
    return img, lung_mask, potential_nodule, regions, labels
```

### Understanding the Logic

**Why `img > -100` and `img < 200`?**
- Normal lung tissue: around -500 HU (dark)
- Nodules: denser, closer to soft tissue (0 to +100 HU)
- We look for "bright spots" inside the normally dark lungs

**The `~` (NOT) operator:**
```python
# ~ inverts True/False
mask = np.array([True, False, True])
print(~mask)  # → [False, True, False]

# Used here to remove LARGE objects:
# 1. ~potential_nodule inverts the mask
# 2. remove_small_objects removes small regions of the INVERSE
# 3. ~(...) inverts back
# Net effect: removes large objects from original mask
```

### Calculating Nodule Size

```python
for r in regions:
    # Estimate diameter from area
    # If area is circular: area = π * r²
    # So diameter ≈ √area * pixel_spacing
    diameter_mm = (r.area ** 0.5) * 0.7  # 0.7 is pixel spacing in mm
    
    print(f"Diameter: {diameter_mm:.1f}mm")
```

**`r.area ** 0.5`** is the same as `sqrt(r.area)` (square root)

**`:.1f`** in f-strings means "1 decimal place, float":
```python
value = 3.14159
print(f"{value:.1f}")  # → 3.1
print(f"{value:.2f}")  # → 3.14
print(f"{value:.0f}")  # → 3
```

---

## 7. Visualization with Matplotlib

### Basic Image Display

```python
import matplotlib.pyplot as plt

# Show single image
plt.imshow(image, cmap='gray')  # cmap = colormap
plt.title('My CT Image')
plt.axis('off')  # Hide axis numbers
plt.show()
```

### Multiple Subplots

```python
# Create figure with 2 rows × 3 columns = 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#                        ↑  ↑          ↑
#                      rows cols    size in inches

# axes is a 2D array of subplot objects
# axes[0][0] = top-left
# axes[0][1] = top-middle
# axes[1][2] = bottom-right

# Loop through and fill each subplot
indices = [50, 100, 150, 200, 250, 300]
for ax, idx in zip(axes.flatten(), indices):
    ax.imshow(windowed[idx], cmap='gray')
    ax.set_title(f'Slice {idx}')
    ax.axis('off')

plt.tight_layout()  # Adjust spacing
plt.savefig('output.png', dpi=150)  # Save to file
plt.show()  # Display on screen
```

### Understanding `zip()` and `flatten()`

```python
# zip combines two lists element-by-element
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age}")
# Output:
# Alice is 25
# Bob is 30
# Charlie is 35

# flatten() converts 2D array to 1D
axes_2d = [[ax1, ax2, ax3],
           [ax4, ax5, ax6]]
axes_1d = axes_2d.flatten()  # → [ax1, ax2, ax3, ax4, ax5, ax6]
```

### Overlaying Images

```python
# Show lung mask overlay on CT
axes[2].imshow(original, cmap='gray')        # Base image
axes[2].imshow(mask, cmap='Reds', alpha=0.3) # Overlay with transparency
#                              ↑
#                          transparency: 0 = invisible, 1 = solid
```

### Drawing Annotations

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')

# Draw a circle
circle = plt.Circle(
    (x, y),           # center position (x, y) NOT (row, col)!
    radius=8,         # radius in pixels
    color='yellow',   # border color
    fill=False,       # don't fill the circle
    linewidth=1.5     # border thickness
)
ax.add_patch(circle)

# Add text label
ax.text(
    x + 10, y,        # position (offset from circle)
    '5.2mm',          # the text
    color='yellow',
    fontsize=7,
    fontweight='bold'
)

plt.show()
```

**Important:** Matplotlib uses (x, y) coordinates where:
- x = column (horizontal)
- y = row (vertical)

But NumPy/image coordinates are (row, col):
- row = y (vertical)
- col = x (horizontal)

So when plotting centroids:
```python
y, x = region.centroid  # centroid returns (row, col)
plt.Circle((x, y), ...)  # Circle wants (x, y)
```

---

## 8. Complete Code Reference

Here's the entire workflow in one place with full comments:

```python
# === IMPORTS ===
import pydicom              # Read DICOM medical images
import numpy as np          # Array operations
import matplotlib.pyplot as plt  # Visualization
from pathlib import Path    # File path handling
from skimage import measure, morphology  # Image processing
import scipy.ndimage as ndimage  # Binary operations


# === LOAD DATA ===
DICOM_FOLDER = "data/patient_001/CT-SCAN"

slices = []
for f in sorted(Path(DICOM_FOLDER).glob("*.dcm")):
    ds = pydicom.dcmread(str(f))
    slices.append(ds)

slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))


# === CONVERT TO HOUNSFIELD UNITS ===
def to_hounsfield(slices):
    volume = np.stack([s.pixel_array.astype(np.int32) for s in slices])
    slope = float(slices[0].RescaleSlope)
    intercept = float(slices[0].RescaleIntercept)
    return volume * slope + intercept

hu_volume = to_hounsfield(slices)


# === APPLY LUNG WINDOW ===
def apply_lung_window(volume):
    return np.clip(volume, -1350, 150)

windowed = apply_lung_window(hu_volume)


# === SEGMENT LUNGS ===
def segment_lungs(hu_volume, slice_idx):
    img = hu_volume[slice_idx]
    
    body_mask = img > -700
    body_filled = ndimage.binary_fill_holes(body_mask)
    lungs_inside = body_filled & (img < -400)
    lungs_clean = morphology.remove_small_objects(lungs_inside, min_size=2000)
    lungs_final = ndimage.binary_fill_holes(lungs_clean)
    
    return img, lungs_final


# === DETECT NODULES ===
def detect_nodule(hu_volume, slice_idx):
    img = hu_volume[slice_idx]
    
    # Segment lungs first
    body_mask = img > -700
    body_filled = ndimage.binary_fill_holes(body_mask)
    lungs_inside = body_filled & (img < -400)
    lungs_clean = morphology.remove_small_objects(lungs_inside, min_size=2000)
    lung_mask = ndimage.binary_fill_holes(lungs_clean)
    
    # Find dense regions inside lungs
    potential_nodule = lung_mask & (img > -100) & (img < 200)
    potential_nodule = morphology.remove_small_objects(potential_nodule, min_size=10)
    potential_nodule = ~morphology.remove_small_objects(~potential_nodule, min_size=2000)
    
    labels = measure.label(potential_nodule)
    regions = measure.regionprops(labels)
    
    return img, lung_mask, potential_nodule, regions, labels


# === VISUALIZE RESULTS ===
slice_idx = 150
img, lung_mask, nodules, regions, labels = detect_nodule(hu_volume, slice_idx)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original CT Slice')
axes[0].axis('off')

axes[1].imshow(img, cmap='gray')
axes[1].imshow(lung_mask, cmap='Blues', alpha=0.3)
axes[1].imshow(nodules, cmap='Reds', alpha=0.6)
axes[1].set_title('Lung Mask + Nodule Candidates')
axes[1].axis('off')

axes[2].imshow(img, cmap='gray')
for r in regions:
    y, x = r.centroid
    diameter_mm = (r.area ** 0.5) * 0.7
    circle = plt.Circle((x, y), radius=8, color='yellow', fill=False)
    axes[2].add_patch(circle)
    axes[2].text(x + 10, y, f'{diameter_mm:.1f}mm', color='yellow', fontsize=7)
axes[2].set_title(f'Detected Nodules: {len(regions)} found')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/nodule_detection.png', dpi=150)
plt.show()
```

---

## 🎯 Quick Reference Cheat Sheet

| Syntax | Meaning | Example |
|--------|---------|---------|
| `x = 5` | Assign value | Store 5 in x |
| `x == 5` | Equals? | Returns True/False |
| `x > 5` | Greater than? | Returns True/False |
| `x & y` | AND (arrays) | Both must be True |
| `x \| y` | OR (arrays) | Either can be True |
| `~x` | NOT (array) | Inverts True/False |
| `x ** 2` | Power | x squared |
| `x ** 0.5` | Square root | √x |
| `[x for x in list]` | List comprehension | Create list from loop |
| `f"text {var}"` | F-string | Insert variable |
| `{x:.2f}` | Format float | 2 decimal places |
| `lambda x: x*2` | Anonymous function | Quick inline function |
| `arr[0]` | First element | Index starts at 0 |
| `arr[-1]` | Last element | Negative indexes |
| `arr[1:4]` | Slice | Elements 1,2,3 |

---

## 📖 Further Learning

1. **Python Basics**: [Python.org Tutorial](https://docs.python.org/3/tutorial/)
2. **NumPy**: [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
3. **Matplotlib**: [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)
4. **Medical Imaging**: [Radiopaedia - Hounsfield Units](https://radiopaedia.org/articles/hounsfield-unit)

---

## 🔬 The Pipeline Visual Summary

```
┌─────────────────┐
│  DICOM Files    │ ← 324 .dcm files from CT scanner
└────────┬────────┘
         │ pydicom.dcmread()
         ▼
┌─────────────────┐
│  Raw Pixels     │ ← uint16 values (0-65535)
└────────┬────────┘
         │ × slope + intercept
         ▼
┌─────────────────┐
│ Hounsfield Units│ ← int32 values (-1024 to +1950)
└────────┬────────┘
         │ np.clip()
         ▼
┌─────────────────┐
│ Windowed Image  │ ← Enhanced for viewing
└────────┬────────┘
         │ Segmentation algorithm
         ▼
┌─────────────────┐
│   Lung Mask     │ ← Binary: True where lung is
└────────┬────────┘
         │ Density thresholding
         ▼
┌─────────────────┐
│ Nodule Candidates│ ← Potential areas of concern
└────────┬────────┘
         │ measure.regionprops()
         ▼
┌─────────────────┐
│ Measurements    │ ← Size, location of each nodule
└─────────────────┘
```

---

Happy learning! 🚀 Feel free to experiment with the code and modify values to see what happens.
