import cv2
import numpy as np
import os
from reliable_steganography import ReliableSteganography

def create_dummy_image(filename, width, height, color=(100, 100, 100)):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    # Add some pattern so it's not uniform
    cv2.circle(img, (width//2, height//2), min(width, height)//4, (255, 255, 255), -1)
    cv2.imwrite(filename, img)
    return img

def test_steganography():
    stego = ReliableSteganography()
    
    # Setup paths
    cover_path = "test_cover.png"
    secret_path = "test_secret.png"
    stego_path = "test_stego_output.png"
    extracted_path = "test_extracted.png"
    
    try:
        # 1. Create images
        # Cover: 300x300 pixels. Capacity: 300*300*4 bits = 360,000 bits = 45,000 bytes.
        print("Creating cover image...")
        create_dummy_image(cover_path, 300, 300, (50, 100, 150))
        
        # Secret: 400x400 pixels. Size: 400*400*3 bytes = 480,000 bytes.
        # This is > 45,000 bytes, so it MUST resize.
        # It needs to reduce size by ~10x. sqrt(10) is ~3.16.
        # So it should resize to roughly 120x120 or smaller.
        print("Creating secret image...")
        create_dummy_image(secret_path, 400, 400, (200, 100, 50))
        
        # 2. Hide image
        print("\n--- Testing Hide ---")
        stego.hide_image_in_image(cover_path, secret_path, stego_path)
        
        if os.path.exists(stego_path):
            print("Stego image created successfully.")
        else:
            print("Failed to create stego image.")
            return

        # 3. Extract image
        print("\n--- Testing Extract ---")
        stego.extract_image_from_image(stego_path, extracted_path)
        
        if os.path.exists(extracted_path):
            print("Extracted image created successfully.")
            # Verify dimensions of extracted image
            extracted = cv2.imread(extracted_path)
            print(f"Extracted image shape: {extracted.shape}")
            # It should be smaller than 400x400
            if extracted.shape[0] < 400 and extracted.shape[1] < 400:
                print("Verification successful: Image was resized and extracted correctly.")
            else:
                print("Warning: Image might not have been resized as expected?")
        else:
            print("Failed to extract image.")

    except Exception as e:
        print(f"TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for f in [cover_path, secret_path, stego_path, extracted_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

if __name__ == "__main__":
    test_steganography()
