import cv2
import numpy as np
import os

# Paths to your already-downloaded model files
prototxt_path = r"C:\Users\shubh\OneDrive\Desktop\12\Colorize_Black_white images\models\colorization_deploy_v2.prototxt"
model_path = r"C:\Users\shubh\OneDrive\Desktop\12\Colorize_Black_white images\models\colorization_release_v2.caffemodel"
pts_path = r"C:\Users\shubh\OneDrive\Desktop\12\Colorize_Black_white images\models\pts_in_hull.npy"

# Check if model files exist
if not all(os.path.exists(f) for f in [prototxt_path, model_path, pts_path]):
    print("Error: Model files missing! Please download them and place in 'models' folder.")
    exit()

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(pts_path)

# Prepare the network
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, np.float32)]

# Load image
image_path = 'girl.jpg'
if not os.path.exists(image_path):
    print(f"Error: Image {image_path} not found!")
    exit()

bw_image = cv2.imread(image_path)

# Resize input image to bigger size (e.g., 512x512)
big_bw = cv2.resize(bw_image, (512, 512))
normalized = big_bw.astype("float32") / 255.0

# Convert to Lab and process
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
L = cv2.split(lab)[0]
L -= 50  # Subtract mean

# Colorize
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Post-processing
ab = cv2.resize(ab, (L.shape[1], L.shape[0]))
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = (255.0 * colorized).astype("uint8")

# ---- ENHANCE COLORS ----
hsv = cv2.cvtColor(colorized, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
s = cv2.multiply(s, 1.5)  # Increase saturation
s = np.clip(s, 0, 255)
hsv_enhanced = cv2.merge([h, s, v])
colorized_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

# ---- BLEND WITH ORIGINAL B/W FOR REALISM ----
bw_float = big_bw.astype(np.float32) / 255.0
color_float = colorized_enhanced.astype(np.float32) / 255.0
final_colorized = cv2.addWeighted(color_float, 0.7, bw_float, 0.3, 0)
final_colorized = (final_colorized * 255).astype(np.uint8)

# ---- FURTHER ENHANCE REALISM USING CLAHE ----
lab_final = cv2.cvtColor(final_colorized, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_final)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l = clahe.apply(l)
lab_enhanced = cv2.merge([l, a, b])
final_colorized_realistic = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# Display results
cv2.imshow("Original", big_bw)
cv2.imshow("Colorized Realistic", final_colorized_realistic)
cv2.waitKey(0)
cv2.destroyAllWindows()
