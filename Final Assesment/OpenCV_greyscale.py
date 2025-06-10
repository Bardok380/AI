import cv2

# Load the image
image = cv2.imread('Images/Lilies.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: image not found of failed to load.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display both original and grayscale images
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)

    # Wait until a key is pressed, then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
