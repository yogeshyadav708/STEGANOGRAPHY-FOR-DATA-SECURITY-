import cv2
import numpy as np
import os
from reliable_steganography import ReliableSteganography

# Create test images
print('Creating test images...')
cover_img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
secret_img = np.random.randint(0, 256, (50, 75, 3), dtype=np.uint8)

cv2.imwrite('cover_test.png', cover_img)
cv2.imwrite('secret_test.png', secret_img)

# Test hide and extract
stego = ReliableSteganography()
print('Testing image-in-image steganography...')

try:
    # Hide image
    success = stego.hide_image_in_image('cover_test.png', 'secret_test.png', 'stego_test.png')
    print(f'Hide operation: {"SUCCESS" if success else "FAILED"}')

    # Extract image
    success = stego.extract_image_from_image('stego_test.png', 'extracted_test.png')
    print(f'Extract operation: {"SUCCESS" if success else "FAILED"}')

    # Verify extracted image exists and has reasonable size
    if os.path.exists('extracted_test.png'):
        extracted = cv2.imread('extracted_test.png')
        if extracted is not None:
            print(f'Extracted image shape: {extracted.shape}')
            print('TEST PASSED: No reshape error occurred!')
        else:
            print('TEST FAILED: Could not read extracted image')
    else:
        print('TEST FAILED: Extracted image file not created')

except Exception as e:
    print(f'TEST FAILED: Error occurred: {str(e)}')

# Cleanup
for f in ['cover_test.png', 'secret_test.png', 'stego_test.png', 'extracted_test.png']:
    if os.path.exists(f):
        os.remove(f)
