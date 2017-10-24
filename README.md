# OpenCVDetectLines
Find the center of a ferris wheel from a night photo.

The pipeline used in this project is as follows:
1. Start from an image, resize it to 800x600
2. Find contours, using Canny (includes a thresholding step)
3. Draw the contours on a new image
4. Erode the contours to reduce thickness (reduces duplicate lines found in the next step)
5. Detect lines on the contour image
6. Draw the detected lines on a new image (for output to user)
7. Find the equations of each line, and store in matrix form
8. Solve the overdetermined system of equations (least squares) to get the optimal intersection point for the lines.  This is the center of the ferris wheel
9. Draw the detected center on the contour image and the original image, and write the coordinates to stdout
