# Computer Vision Assignment

## Question 1  
### Use computer vision techniques to detect, segment, and count coins from an image containing scattered Indian coins.  

---

### a. Detect all coins in the image (2 Marks)  
- Use edge detection to detect all coins in the image.  
- Visualize the detected coins by outlining them in the image.  

### b. Segmentation of Each Coin (3 Marks)  
- Apply region-based segmentation techniques to isolate individual coins from the image.  
- Provide segmented outputs for each detected coin.  

### c. Count the Total Number of Coins (2 Marks)  
- Write a function to count the total number of coins detected in the image.  
- Display the final count as an output.  

**Input Data:**  
Capture or obtain your own image containing various Indian coins.  

---

## Implementation Details  

1. **Grayscale Conversion & Thresholding**  
   - The input image was converted to grayscale.  
   - Otsu's thresholding was applied to create a binary image.  
   - *(Optional step)* The binary image was inverted but can be skipped.  

2. **Background Modification & Segmentation**  
   - A completely white image of the same size as the grayscale image was created.  
   - Pixels corresponding to the background in the binary image were set to white in the grayscale image.  
   - This effectively segmented the objects from the background.  

3. **Edge Detection & Contour Enhancement**  
   - Canny edge detection was applied to the white-background image.  
   - Edge thickening was performed to enhance coin contours while suppressing weaker edges inside the coins, reducing noise.  
   - The thickened edges were then used for individual coin segmentation using Otsu's thresholding.  

---

## Outputs  

### a1) Canny Edge Detection Output (After Thickening Edges)  
![Canny Edge Detection](https://github.com/user-attachments/assets/e18386e4-bdf4-486a-9588-ecde617cdbd0)  

### a2) Original Image with Detected Coins Outlined  
![Detected Coins](https://github.com/user-attachments/assets/0c68d73d-32d1-4d08-adde-0022116e3e55)  

### b) Segmented Individual Coins  
![Segmented Coins](https://github.com/user-attachments/assets/1f88a09a-23fb-484e-bdac-e6853eb0c781)  

### c) Total Count of Coins  
**Output:** 8 Coins  

---

## Additional Experiments  

Two alternative approaches were tested:  
1. Marr-Hildreth Edge Detector  
2. Canny Edge Detection (Without Thickening Edges)  

Both approaches were found to be less effective, as they missed detecting one or more coins.  
The respective codes are commented out but can be uncommented to observe their outputs.  

---

# Question 2  
### Create a stitched panorama from multiple overlapping images.  

---

### a. Extract Key Points (1 Mark)  
- Detect key points in overlapping images.  

### b. Image Stitching (2 Marks)  
- Use the extracted key points to align and stitch the images into a single panorama.  
- Provide the final panorama image as output.  

**Dataset:**  
Capture your own set of overlapping images using a smartphone.  

---

## Implementation Details  

1. **Feature Detection using SIFT**  
   - SIFT (Scale-Invariant Feature Transform) was used to detect key points in the images.  

2. **Pairwise Image Stitching**  
   - Key points were matched for every consecutive pair of images from a set of 7 images.  
   - Each pair was stitched together progressively to form the final panorama.  

3. **Comparison with OpenCV’s Default Stitcher**  
   - OpenCV’s built-in stitcher was also tested, and the output can be obtained by running the code.  
   - However, the OpenCV stitched output is not included here.  

---

## Output  

### Final Panorama Image  
![Panorama Image](https://github.com/user-attachments/assets/c9a03e63-19b7-4fc6-8225-2e13dd9e1c69)  

---

## Dependencies  

The following Python libraries are required to run the project:  

- `cv2` (OpenCV)  
- `numpy`  
- `matplotlib.pyplot`  

To install them, run:  

```bash
pip install opencv-python numpy matplotlib
