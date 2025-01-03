import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


def det(key):
    """Check if the determinant of the key matrix is valid."""
    a, b = key[0]
    c, d = key[1]
    determinant = (a * d) - (b * c)
    return determinant % 256 == 1 and determinant != 0


def modular_inverse(a, mod):
    """Compute the modular inverse of a under mod."""
    for x in range(1, mod):
        if (a * x) % mod == 1:
            return x
    return None


def enc(img, key, cho):
    """Encrypt or decrypt the image using the given key."""
    if not det(key):
        return "Matrix is invalid. Ensure det(key) % 256 == 1 and key is invertible."

    if cho == 1:
        a, b = key[0]
        c, d = key[1]
    elif cho == 2:
        determinant = (key[0][0] * key[1][1] - key[0][1] * key[1][0]) % 256
        inv_det = modular_inverse(determinant, 256)
        if inv_det is None:
            return "Matrix is not invertible in mod 256."
        a, b = (key[1][1] * inv_det) % 256, (-key[0][1] * inv_det) % 256
        c, d = (-key[1][0] * inv_det) % 256, (key[0][0] * inv_det) % 256
    else:
        return "Invalid choice (cho). Use 1 for encoding, 2 for decoding."

    result = np.zeros_like(img)
    rows, cols, channels = img.shape
    for j in range(0, rows - 1, 2):
        for k in range(cols):
            for channel in range(channels):
                r1 = ((img[j, k, channel] * a) + (img[j + 1, k, channel] * c)) % 256
                r2 = ((img[j, k, channel] * b) + (img[j + 1, k, channel] * d)) % 256

                result[j, k, channel] = r1
                result[j + 1, k, channel] = r2

    # Handle the last row if the number of rows is odd
    if rows % 2 == 1:
        result[-1] = img[-1]

    return result.astype(np.uint8)


# Debugging the Image Loading
img_path = r"C:\Users\hp\Documents\SMESTER 5\Pengolahan Citra Digital\sesi 14\saskehhhh.jpeg"

try:
    img = iio.imread(img_path)
    print("Image loaded successfully.")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Image must be in RGB format.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Debugging the Key
key = np.array([
    [5, 3],
    [3, 2]
])
if not det(key):
    print("Invalid key matrix. Ensure determinant % 256 == 1.")
    exit()

# Encode and Decode the Image
imgEnc = enc(img, key, 1)
if isinstance(imgEnc, str):  # Check for errors
    print(imgEnc)
    exit()

imgDec = enc(imgEnc, key, 2)
if isinstance(imgDec, str):  # Check for errors
    print(imgDec)
    exit()

# Plot the Images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Encoded Image")
plt.imshow(imgEnc)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Decoded Image")
plt.imshow(imgDec)
plt.axis("off")

plt.tight_layout()
plt.show()
