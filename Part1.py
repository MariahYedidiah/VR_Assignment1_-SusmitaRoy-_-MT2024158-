import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert background to white
    background_mask = cv2.bitwise_not(binary)
    white_background = np.full_like(gray, 255)
    segmented = np.where(background_mask == 255, white_background, gray)
    
    return image, segmented

def detect_edges(segmented):
    # Apply Canny Edge Detection
    edges = cv2.Canny(segmented, 100, 200)
    
    # Apply Morphological Dilation to thicken edges
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    
    return thick_edges

def find_and_segment_coins(image, thick_edges):
    # Find contours
    contours, _ = cv2.findContours(thick_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coin_count = 0
    segmented_coins = []
    result_image = image.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter small regions
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
            
            if 0.7 < circularity < 1.3:  # Consider circular shapes
                coin_count += 1
                cv2.drawContours(result_image, [cnt], -1, (0, 255, 0), 2)  # Draw contour
                
                # Get bounding box around the coin
                x, y, w, h = cv2.boundingRect(cnt)
                coin_roi = image[y:y+h, x:x+w]  # Crop the coin
                segmented_coins.append(coin_roi)
    
    return result_image, segmented_coins, coin_count

def display_results(thick_edges, result_image, segmented_coins, coin_count):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(thick_edges, cmap="gray")
    axs[0].set_title("Canny Edge Detection")
    axs[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Segmented Coins - Count: {coin_count}")
    
    for ax in axs.flat:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Display Each Segmented Coin Individually
    fig, axs = plt.subplots(1, len(segmented_coins), figsize=(15, 5))
    for i, coin in enumerate(segmented_coins):
        axs[i].imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Coin {i+1}")
        axs[i].axis("off")
    
    plt.show()
    
    print(f"Total number of detected coins: {coin_count}")

# Main Execution
image_path = "Images/coins2.jpeg"
image, segmented = preprocess_image(image_path)
thick_edges = detect_edges(segmented)
result_image, segmented_coins, coin_count = find_and_segment_coins(image, thick_edges)
display_results(thick_edges, result_image, segmented_coins, coin_count)




"""MARR-HILDRETH"""
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = "Images/coins2.jpeg"  # Update with your image path
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # **1. Segmentation - Remove Background**
# _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# background_mask = cv2.bitwise_not(binary)
# white_background = np.full_like(gray, 255)
# segmented = np.where(background_mask == 255, white_background, gray)

# # **2. Apply Gaussian Blur to Reduce Noise**
# blurred = cv2.GaussianBlur(segmented, (5, 5), 0)

# # **3. Apply the Laplacian of Gaussian (LoG) - Marr-Hildreth Edge Detection**
# laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
# laplacian_abs = np.uint8(np.absolute(laplacian))  # Convert to uint8

# # **4. Normalize and Threshold the Laplacian Output**
# _, edges = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)

# # **5. Thicken Edges Using Morphological Dilation**
# kernel = np.ones((3, 3), np.uint8)  # Kernel size determines thickness
# thick_edges = cv2.dilate(edges, kernel, iterations=1)  # One iteration for slight thickening

# # **6. Find Contours**
# contours, _ = cv2.findContours(thick_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # **7. Filter Circular Contours and Count Coins**
# coin_count = 0
# result_image = image.copy()
# mask = np.zeros_like(gray)  # Mask for blurring inside coins

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > 500:  # Ignore small noise
#         perimeter = cv2.arcLength(cnt, True)
#         circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
#         if 0.7 < circularity < 1.3:  # Consider nearly circular shapes
#             coin_count += 1
#             cv2.drawContours(result_image, [cnt], -1, (0, 255, 0), 2)  # Green contour around coins
            
#             # Fill the detected coin in mask to apply blur inside it
#             # cv2.drawContours(mask, [cnt], -1, 255, -1)

# # **8. Blur Inside Coins**
# #blurred_inside = cv2.GaussianBlur(result_image, (15, 15), 10)  # Apply heavy blur
# # result_blurred = np.where(mask[:, :, None] == 255, blurred_inside, result_image)  # Merge blurred areas

# # **9. Display Results**
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# axs[0].imshow(thick_edges, cmap="gray")
# axs[0].set_title("Thickened Marr-Hildreth Edges")

# # axs[1].imshow(cv2.cvtColor(result_blurred, cv2.COLOR_BGR2RGB))
# # axs[1].set_title(f"Detected Coins - Blurred Inside (Count: {coin_count})")

# axs[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
# axs[1].set_title(f"Detected Coins - Without Blur")

# for ax in axs.flat:
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# print(f"Total number of detected coins: {coin_count}")








"""CANNY-WITHOUT THICKENING"""
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = "Images/coins2.jpeg"  # Update this if needed
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # **1. Apply Otsu's Thresholding for Segmentation (No Background Change)**
# _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # **2. Apply Canny Edge Detection**
# edges = cv2.Canny(gray, 50, 150)

# # **3. Slightly Thicker Canny Edges (Less Overlap)**
# kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
# thick_edges = cv2.dilate(edges, kernel, iterations=1)  # Only 1 iteration

# # **4. Find Contours from Thick Edges**
# contours, _ = cv2.findContours(thick_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # **5. Filter Circular Contours and Count Coins**
# coin_count = 0
# result_image = image.copy()
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > 500:  # Filter small regions
#         perimeter = cv2.arcLength(cnt, True)
#         circularity = (4 * np.pi * area) / (perimeter * perimeter)
#         if 0.7 < circularity < 1.3:  # Consider circular shapes
#             coin_count += 1
#             cv2.drawContours(result_image, [cnt], -1, (0, 255, 0), 2)  # Green contour

# # **6. Display Results in the Same Window**
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# axs[0].imshow(thick_edges, cmap="gray")
# axs[0].set_title("Thick Canny Edge Detection")

# axs[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
# axs[1].set_title(f"Segmented Coins - Count: {coin_count}")

# for ax in axs.flat:
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

# print(f"Total number of detected coins: {coin_count}")
