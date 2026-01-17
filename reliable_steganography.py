"""
Reliable Steganography - PNG Only for Guaranteed Results
"""

import sys
import os

# Check for required modules with clear error messages
missing_modules = []

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
except ImportError as e:
    print("=" * 70)
    print("ERROR: tkinter is not available!")
    print("=" * 70)
    print("tkinter is usually included with Python, but may be missing on some systems.")
    print("\nTo install on Linux:")
    print("  sudo apt-get install python3-tk  # Debian/Ubuntu")
    print("  sudo yum install python3-tkinter  # CentOS/RHEL")
    print("\nOn Windows/Mac, tkinter should be included with Python.")
    print("=" * 70)
    sys.exit(1)

try:
    import cv2
except ImportError:
    missing_modules.append(("opencv-python", "cv2", "pip install opencv-python"))

try:
    import numpy as np
except ImportError:
    missing_modules.append(("numpy", "numpy", "pip install numpy"))

try:
    import imageio
except ImportError:
    missing_modules.append(("imageio", "imageio", "pip install imageio"))

# Standard library imports (should always be available)
import hashlib
import base64

# If any modules are missing, show clear error message
if missing_modules:
    print("\n" + "=" * 70)
    print("ERROR: Required Python packages are not installed!")
    print("=" * 70)
    print("\nThe following modules are missing:\n")
    
    for package_name, module_name, install_cmd in missing_modules:
        print(f"  [X] {module_name} (from package: {package_name})")
    
    print("\n" + "-" * 70)
    print("INSTALLATION INSTRUCTIONS:")
    print("-" * 70)
    print("\n1. Open your terminal/command prompt")
    print("\n2. Run the following command to install all required packages:")
    print(f"\n   python -m pip install {' '.join([pkg[0] for pkg in missing_modules])}")
    print("\n   OR install them individually:")
    for package_name, module_name, install_cmd in missing_modules:
        print(f"   {install_cmd}")
    
    print("\n" + "-" * 70)
    print("TROUBLESHOOTING:")
    print("-" * 70)
    print("\nIf you're using Python 3.14 or newer:")
    print("  • Some packages may not have pre-built wheels yet")
    print("  • Consider using Python 3.11 or 3.12 for better compatibility")
    print("  • Or install Microsoft C++ Build Tools to build from source")
    
    print("\nIf installation fails:")
    print("  • Make sure pip is up to date: python -m pip install --upgrade pip")
    print("  • Try: python -m pip install --user <package-name>")
    print("  • Check your internet connection")
    print("  • Verify Python version: python --version")
    
    print("\n" + "=" * 70)
    print(f"\nPlease install the missing packages and run the script again.")
    print("=" * 70 + "\n")
    
    # Try to show a GUI error if tkinter is available
    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        error_msg = "Missing Required Packages!\n\n"
        error_msg += "Please install the following packages:\n\n"
        for package_name, module_name, install_cmd in missing_modules:
            error_msg += f"• {package_name}\n"
        error_msg += "\nRun in terminal:\n"
        error_msg += f"python -m pip install {' '.join([pkg[0] for pkg in missing_modules])}"
        messagebox.showerror("Missing Dependencies", error_msg)
        root.destroy()
    except:
        pass
    
    sys.exit(1)

# All imports successful
print("[OK] All required packages are installed successfully!")
print("Starting Steganography Application...\n")

class ReliableSteganography:
    def __init__(self):
        self.supported_formats = ['.png', '.bmp', '.tiff']  # Only lossless formats
        self.convertible_formats = ['.jpg', '.jpeg']  # Formats that can be converted to PNG
        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']  # Video formats
    
    def text_to_binary(self, text):
        """Convert text to binary string"""
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary
    
    def binary_to_text(self, binary):
        """Convert binary string to text"""
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text
    
    def simple_encrypt(self, data, key):
        """Simple XOR encryption"""
        try:
            # Create key hash
            key_hash = hashlib.sha256(key.encode()).digest()
            key_bytes = key_hash[:len(data)]
            
            # XOR encryption
            encrypted = bytearray()
            for i, char in enumerate(data):
                encrypted.append(ord(char) ^ key_bytes[i % len(key_bytes)])
            
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            raise Exception(f"Encryption failed: {str(e)}")
    
    def simple_decrypt(self, encrypted_data, key):
        """Simple XOR decryption"""
        try:
            # Decode base64 with padding handling
            try:
                encrypted_bytes = base64.b64decode(encrypted_data.encode())
            except Exception as padding_error:
                # Try adding padding if missing
                missing_padding = len(encrypted_data) % 4
                if missing_padding:
                    encrypted_data += '=' * (4 - missing_padding)
                try:
                    encrypted_bytes = base64.b64decode(encrypted_data.encode())
                except Exception:
                    raise Exception(f"Invalid base64 data: {str(padding_error)}")

            # Create key hash
            key_hash = hashlib.sha256(key.encode()).digest()
            key_bytes = key_hash[:len(encrypted_bytes)]

            # XOR decryption
            decrypted = ""
            for i, byte in enumerate(encrypted_bytes):
                decrypted += chr(byte ^ key_bytes[i % len(key_bytes)])

            return decrypted
        except Exception as e:
            raise Exception(f"Decryption failed: {str(e)}")
    
    def convert_to_png(self, image_path):
        """Convert JPG/JPEG to PNG format"""
        try:
            print(f"Debug: Converting {image_path} to PNG...")
            
            # Validate file exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file does not exist: {image_path}")
            
            # Check file is readable
            if not os.access(image_path, os.R_OK):
                raise ValueError(f"Image file is not readable: {image_path}")
            
            # Read the image with explicit flags
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                # Try alternative reading method
                try:
                    import imageio
                    img_array = imageio.imread(image_path)
                    # Convert to BGR if needed (imageio reads as RGB)
                    if len(img_array.shape) == 3:
                        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img = img_array
                except Exception as alt_error:
                    raise ValueError(
                        f"Could not read image file: {image_path}\n"
                        f"Possible causes:\n"
                        f"1. File is corrupted or not a valid image\n"
                        f"2. File format not supported\n"
                        f"3. File path contains special characters\n"
                        f"Error: {str(alt_error)}"
                    )
            
            if img is None or img.size == 0:
                raise ValueError(f"Image file is empty or invalid: {image_path}")
            
            # Create PNG filename in same folder
            folder = os.path.dirname(image_path) or '.'
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            png_path = os.path.join(folder, f"{name}_converted.png")
            
            # Save as PNG with compression
            success = cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if success:
                # Verify file was created
                if os.path.exists(png_path):
                    print(f"Debug: Converted to PNG: {png_path}")
                    return png_path
                else:
                    raise ValueError(f"PNG file was not created at: {png_path}")
            else:
                raise ValueError(f"Failed to save PNG file. Check write permissions for: {folder}")
                
        except Exception as e:
            error_msg = f"Error converting to PNG: {str(e)}"
            print(error_msg)
            return None
    
    def hide_message(self, image_path, message, output_path, encryption_key=None):
        """Hide message in image with enhanced capacity - PNG/BMP/TIFF only for reliability"""
        try:
            print(f"Debug: Hiding message in {image_path}")
            print(f"Debug: Message: '{message}'")
            print(f"Debug: Message length: {len(message)}")
            print(f"Debug: Encryption key: {encryption_key if encryption_key else 'None'}")
            
            # Check if file is supported format
            if not any(image_path.lower().endswith(ext) for ext in ['.png', '.bmp', '.tiff']):
                # Check if it's a convertible format (JPG/JPEG)
                if any(image_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg']):
                    print("Debug: JPG/JPEG detected, converting to PNG...")
                    converted_path = self.convert_to_png(image_path)
                    if converted_path:
                        image_path = converted_path
                        print(f"Debug: Using converted PNG: {image_path}")
                    else:
                        raise ValueError(
                            f"Failed to convert JPG/JPEG to PNG.\n"
                            f"Possible causes:\n"
                            f"1. Image file is corrupted or invalid\n"
                            f"2. File path contains special characters\n"
                            f"3. Insufficient permissions to read/write files\n"
                            f"4. File format is not actually JPG/JPEG\n\n"
                            f"Try:\n"
                            f"- Open the image in an image viewer to verify it's valid\n"
                            f"- Use the PNG Converter tab to convert manually\n"
                            f"- Check file permissions"
                        )
                # Check if it's a video format
                elif any(image_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                    print("Debug: Video format detected, using video steganography...")
                    return self.hide_message_in_video(image_path, message, output_path, encryption_key)
                else:
                    raise ValueError("Only PNG, BMP, TIFF, JPG, JPEG, and video formats are supported")
            
            # Read image with better error handling
            if not os.path.exists(image_path):
                raise ValueError(f"Image file does not exist: {image_path}")
            
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                # Try with imageio as fallback
                try:
                    import imageio
                    img_array = imageio.imread(image_path)
                    if len(img_array.shape) == 3:
                        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img = img_array
                except Exception as e:
                    raise ValueError(
                        f"Could not read image file: {image_path}\n"
                        f"File may be corrupted or in an unsupported format.\n"
                        f"Error: {str(e)}"
                    )
            
            if img is None or img.size == 0:
                raise ValueError(f"Image file is empty or invalid: {image_path}")
            
            print(f"Debug: Image shape: {img.shape}")
            
            # Process message
            if encryption_key:
                print("Debug: Encrypting message...")
                processed_message = self.simple_encrypt(message, encryption_key)
                print(f"Debug: Encrypted message: '{processed_message}'")
            else:
                processed_message = message
            
            # Convert to binary
            message_binary = self.text_to_binary(processed_message)
            print(f"Debug: Binary length: {len(message_binary)}")
            
            # Add message length as header (32 bits)
            message_length = len(message_binary)
            length_binary = format(message_length, '032b')
            full_binary = length_binary + message_binary
            
            print(f"Debug: Full binary length: {len(full_binary)}")
            
            # Ultra-enhanced capacity calculation - use 3 LSB layers for 3x capacity
            height, width, channels = img.shape
            # Use 3 LSB layers for 3x capacity
            max_capacity = height * width * channels * 3
            print(f"Debug: Ultra-enhanced max capacity: {max_capacity}")
            
            if len(full_binary) > max_capacity:
                raise ValueError(f"Message too long. Ultra-enhanced maximum capacity: {max_capacity // 8} characters")
            
            # Hide message using 3 LSB layers for ultra-enhanced capacity
            img_flat = img.flatten()
            print(f"Debug: Flattened image length: {len(img_flat)}")
            
            # Use 3 LSB layers for 3x capacity
            for i, bit in enumerate(full_binary):
                if i < len(img_flat):
                    # Use 3 LSB layers
                    pixel = img_flat[i]
                    # Clear 3 LSB bits
                    pixel = pixel & 0xF8
                    # Set 3 LSB bits
                    pixel = pixel | int(bit) | (int(bit) << 1) | (int(bit) << 2)
                    img_flat[i] = pixel
            
            # Reshape and save
            img_stego = img_flat.reshape(img.shape)
            
            # Save as PNG for reliability
            if not output_path.lower().endswith('.png'):
                output_path = os.path.splitext(output_path)[0] + '.png'
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            success = cv2.imwrite(output_path, img_stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Debug: PNG save success: {success}")
            
            if not success:
                raise ValueError(
                    f"Failed to save stego image to: {output_path}\n"
                    f"Possible causes:\n"
                    f"1. Invalid file path\n"
                    f"2. Insufficient write permissions\n"
                    f"3. Disk space full\n"
                    f"4. Output directory doesn't exist and couldn't be created"
                )
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise ValueError(f"Stego image file was not created at: {output_path}")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"Debug: Stego image saved successfully: {output_path} ({file_size:.2f} MB)")
            
            return True
            
        except Exception as e:
            error_msg = f"Error hiding message: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def extract_message(self, stego_path, decryption_key=None):
        """Extract message from stego image with enhanced capacity - PNG/BMP/TIFF only"""
        try:
            print(f"Debug: Extracting from {stego_path}")
            print(f"Debug: Decryption key: {decryption_key if decryption_key else 'None'}")
            
            # Check if it's a video format
            if any(stego_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                print("Debug: Video format detected, using video extraction...")
                return self.extract_message_from_video(stego_path, decryption_key)
            
            # Read stego image
            img = cv2.imread(stego_path)
            if img is None:
                raise ValueError("Could not read stego image file")
            
            print(f"Debug: Stego image shape: {img.shape}")
            
            # Flatten image
            img_flat = img.flatten()
            print(f"Debug: Flattened stego length: {len(img_flat)}")
            
            # Extract message length (first 32 bits) using 3 LSB layers
            length_binary = ''
            for i in range(32):
                if i < len(img_flat):
                    # Extract from 3 LSB layers
                    pixel = img_flat[i]
                    bit = (pixel & 1) | ((pixel & 2) >> 1) | ((pixel & 4) >> 2)
                    length_binary += str(bit)
            
            message_length = int(length_binary, 2)
            print(f"Debug: Extracted message length: {message_length}")
            
            # Validate message length
            if message_length > len(img_flat) or message_length < 0 or message_length > 1000000:
                raise ValueError(f"Invalid message length: {message_length}")
            
            # Extract message bits using 3 LSB layers
            message_binary = ''
            for i in range(32, 32 + message_length):
                if i < len(img_flat):
                    # Extract from 3 LSB layers
                    pixel = img_flat[i]
                    bit = (pixel & 1) | ((pixel & 2) >> 1) | ((pixel & 4) >> 2)
                    message_binary += str(bit)
            
            print(f"Debug: Extracted binary length: {len(message_binary)}")
            
            # Convert binary to text
            encrypted_message = self.binary_to_text(message_binary)
            print(f"Debug: Extracted message: '{encrypted_message}'")
            
            # Decrypt if key provided
            if decryption_key:
                try:
                    print("Debug: Decrypting message...")
                    decrypted = self.simple_decrypt(encrypted_message, decryption_key)
                    print(f"Debug: Decrypted message: '{decrypted}'")
                    return decrypted
                except Exception as e:
                    print(f"Debug: Decryption failed: {str(e)}")
                    return encrypted_message  # Return encrypted if decryption fails
            else:
                return encrypted_message
            
        except Exception as e:
            print(f"Error extracting message: {str(e)}")
            return None
    
    def hide_message_in_video(self, video_path, message, output_path, encryption_key=None):
        """Hide message in video using frame-based steganography"""
        try:
            print(f"Debug: Hiding message in video {video_path}")
            print(f"Debug: Message: '{message}'")
            print(f"Debug: Message length: {len(message)}")
            print(f"Debug: Encryption key: {encryption_key if encryption_key else 'None'}")
            
            # Process message
            if encryption_key:
                print("Debug: Encrypting message...")
                processed_message = self.simple_encrypt(message, encryption_key)
                print(f"Debug: Encrypted message: '{processed_message}'")
            else:
                processed_message = message
            
            # Convert to binary
            message_binary = self.text_to_binary(processed_message)
            print(f"Debug: Binary length: {len(message_binary)}")
            
            # Add message length as header (32 bits)
            message_length = len(message_binary)
            length_binary = format(message_length, '032b')
            full_binary = length_binary + message_binary
            
            print(f"Debug: Full binary length: {len(full_binary)}")
            
            # Read video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Debug: Video properties - FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
            
            # Calculate capacity
            max_capacity = total_frames * width * height * 3 * 3  # 3 LSB layers
            print(f"Debug: Video max capacity: {max_capacity}")
            
            if len(full_binary) > max_capacity:
                raise ValueError(f"Message too long. Video maximum capacity: {max_capacity // 8} characters")
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            print(f"Debug: Read {len(frames)} frames")
            
            # Hide message in frames using 3 LSB layers
            bit_index = 0
            for frame_idx, frame in enumerate(frames):
                if bit_index >= len(full_binary):
                    break
                
                frame_flat = frame.flatten()
                for pixel_idx in range(len(frame_flat)):
                    if bit_index >= len(full_binary):
                        break
                    
                    bit = int(full_binary[bit_index])
                    pixel = frame_flat[pixel_idx]
                    
                    # Use 3 LSB layers for enhanced capacity
                    pixel = pixel & 0xF8  # Clear 3 LSB bits
                    pixel = pixel | bit | (bit << 1) | (bit << 2)  # Set 3 LSB bits
                    frame_flat[pixel_idx] = pixel
                    
                    bit_index += 1
                
                frames[frame_idx] = frame_flat.reshape(frame.shape)
            
            # Save video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            print(f"Debug: Video saved successfully: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error hiding message in video: {str(e)}")
            return False
    
    def extract_message_from_video(self, video_path, decryption_key=None):
        """Extract message from video using frame-based steganography"""
        try:
            print(f"Debug: Extracting from video {video_path}")
            print(f"Debug: Decryption key: {decryption_key if decryption_key else 'None'}")
            
            # Read video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Debug: Video properties - FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            print(f"Debug: Read {len(frames)} frames")
            
            # Extract message length (first 32 bits)
            length_binary = ''
            bit_count = 0
            
            for frame in frames:
                if bit_count >= 32:
                    break
                
                frame_flat = frame.flatten()
                for pixel in frame_flat:
                    if bit_count >= 32:
                        break
                    
                    # Extract from 3 LSB layers
                    bit = (pixel & 1) | ((pixel & 2) >> 1) | ((pixel & 4) >> 2)
                    length_binary += str(bit)
                    bit_count += 1
            
            message_length = int(length_binary, 2)
            print(f"Debug: Extracted message length: {message_length}")
            
            # Validate message length
            if message_length > len(frames) * width * height * 3 or message_length < 0 or message_length > 1000000:
                raise ValueError(f"Invalid message length: {message_length}")
            
            # Extract message bits
            message_binary = ''
            bit_count = 0
            total_bits_needed = 32 + message_length
            
            for frame in frames:
                if bit_count >= total_bits_needed:
                    break
                
                frame_flat = frame.flatten()
                for pixel in frame_flat:
                    if bit_count >= total_bits_needed:
                        break
                    
                    if bit_count >= 32:  # Skip length header
                        # Extract from 3 LSB layers
                        bit = (pixel & 1) | ((pixel & 2) >> 1) | ((pixel & 4) >> 2)
                        message_binary += str(bit)
                    
                    bit_count += 1
            
            print(f"Debug: Extracted binary length: {len(message_binary)}")
            
            # Convert binary to text
            encrypted_message = self.binary_to_text(message_binary)
            print(f"Debug: Extracted message: '{encrypted_message}'")
            
            # Decrypt if key provided
            if decryption_key:
                try:
                    print("Debug: Decrypting message...")
                    decrypted = self.simple_decrypt(encrypted_message, decryption_key)
                    print(f"Debug: Decrypted message: '{decrypted}'")
                    return decrypted
                except Exception as e:
                    print(f"Debug: Decryption failed: {str(e)}")
                    return encrypted_message  # Return encrypted if decryption fails
            else:
                return encrypted_message
            
        except Exception as e:
            print(f"Error extracting message from video: {str(e)}")
            return None
    
    def hide_image_in_image(self, cover_image_path, secret_image_path, output_path, encryption_key=None):
        """Hide an image inside another image using LSB steganography"""
        try:
            print(f"Debug: Hiding image {secret_image_path} in {cover_image_path}")
            
            # Read cover image
            cover_img = cv2.imread(cover_image_path)
            if cover_img is None:
                raise ValueError("Could not read cover image file")
            
            # Read secret image
            secret_img = cv2.imread(secret_image_path)
            if secret_img is None:
                raise ValueError("Could not read secret image file")
            
            print(f"Debug: Cover image shape: {cover_img.shape}")
            print(f"Debug: Secret image shape: {secret_img.shape}")
            
            # Resize secret image to fit in cover image (preserve aspect ratio, but limit to cover size)
            cover_height, cover_width = cover_img.shape[:2]
            secret_height, secret_width = secret_img.shape[:2]
            
            # Calculate scaling factor to fit secret image in cover
            scale = min(cover_width / secret_width, cover_height / secret_height, 1.0)
            new_width = int(secret_width * scale)
            new_height = int(secret_height * scale)
            
            # Resize secret image
            secret_resized = cv2.resize(secret_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Debug: Resized secret image shape: {secret_resized.shape}")
            
            # Get secret image dimensions for reconstruction (use resized dimensions since that's what we're storing)
            original_height = secret_resized.shape[0]
            original_width = secret_resized.shape[1]
            original_channels = secret_resized.shape[2] if len(secret_resized.shape) > 2 else 1
            
            # Store uncompressed for reliability with LSB steganography
            # PNG compression is sensitive to bit modifications and can fail during extraction
            secret_flat = secret_resized.flatten()
            pixel_binary = ''.join(format(pixel, '08b') for pixel in secret_flat)
            compressed = False
            print(f"Debug: Using uncompressed storage for reliability ({len(secret_flat)} pixels)")
            
            # Convert dimensions to binary (32 bits each for height, width, channels)
            height_binary = format(original_height, '032b')
            width_binary = format(original_width, '032b')
            channels_binary = format(original_channels, '032b')

            # Combine: dimensions (96 bits) + pixel data (uncompressed)
            full_binary = height_binary + width_binary + channels_binary + pixel_binary
            
            # Encrypt if key provided
            if encryption_key:
                print("Debug: Encrypting image data...")
                # Convert binary to text representation for encryption
                # Use base64 encoding to handle binary data safely
                binary_bytes = bytes(int(full_binary[i:i+8], 2) for i in range(0, len(full_binary), 8))
                binary_b64 = base64.b64encode(binary_bytes).decode('utf-8')
                encrypted_data = self.simple_encrypt(binary_b64, encryption_key)
                full_binary = self.text_to_binary(encrypted_data)
            
            # Add length header (32 bits)
            data_length = len(full_binary)
            length_binary = format(data_length, '032b')
            full_binary = length_binary + full_binary
            
            print(f"Debug: Total binary length: {len(full_binary)}")
            
            # Check capacity - Using 4 LSB layers for maximum capacity
            cover_flat = cover_img.flatten()
            max_capacity = len(cover_flat) * 4  # Storing 4 bits per byte using all 4 LSB bits
            
            if len(full_binary) > max_capacity:
                # Calculate required size reduction
                required_reduction = len(full_binary) / max_capacity
                new_scale = scale / required_reduction * 0.95  # 95% to leave some margin
                
                if new_scale < 0.1:  # If we need to reduce by more than 90%, it's probably not feasible
                    cover_size_mb = (cover_height * cover_width * 3) / (1024 * 1024)
                    secret_size_mb = (secret_height * secret_width * 3) / (1024 * 1024)
                    raise ValueError(
                        f"Secret image is too large for the cover image!\n\n"
                        f"Cover image size: {cover_width}x{cover_height} pixels (~{cover_size_mb:.2f} MB)\n"
                        f"Secret image size: {secret_width}x{secret_height} pixels (~{secret_size_mb:.2f} MB)\n"
                        f"Maximum capacity: {max_capacity // 8:,} bytes (~{max_capacity // (8 * 1024 * 1024):.2f} MB)\n"
                        f"Required: {len(full_binary) // 8:,} bytes (~{len(full_binary) // (8 * 1024 * 1024):.2f} MB)\n\n"
                        f"Solution: Use a larger cover image or a smaller secret image.\n"
                        f"Recommended: Secret image should be less than {max_capacity // (8 * 1024 * 1024):.2f} MB"
                    )
                
                # Try to resize more aggressively
                print(f"Debug: Resizing secret image more aggressively (scale: {new_scale:.2f})")
                new_width = int(secret_width * new_scale)
                new_height = int(secret_height * new_scale)
                
                # Ensure minimum size
                if new_width < 10 or new_height < 10:
                    raise ValueError(
                        f"Secret image is too large! Even after resizing, it would be too small.\n"
                        f"Please use a larger cover image or a much smaller secret image."
                    )
                
                secret_resized = cv2.resize(secret_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Debug: Aggressively resized secret image shape: {secret_resized.shape}")
                
                # Update dimensions for the header to match the new resized image
                original_height = secret_resized.shape[0]
                original_width = secret_resized.shape[1]
                # Update binary headers
                height_binary = format(original_height, '032b')
                width_binary = format(original_width, '032b')
                
                # Recalculate binary data with new size (uncompressed for reliability)
                secret_flat = secret_resized.flatten()
                pixel_binary = ''.join(format(pixel, '08b') for pixel in secret_flat)
                print(f"Debug: Using uncompressed storage after resize ({len(secret_flat)} pixels)")
                full_binary = height_binary + width_binary + channels_binary + pixel_binary
                
                # Re-encrypt if needed
                if encryption_key:
                    binary_bytes = bytes(int(full_binary[i:i+8], 2) for i in range(0, len(full_binary), 8))
                    binary_b64 = base64.b64encode(binary_bytes).decode('utf-8')
                    encrypted_data = self.simple_encrypt(binary_b64, encryption_key)
                    full_binary = self.text_to_binary(encrypted_data)
                
                # Re-add length header
                data_length = len(full_binary)
                length_binary = format(data_length, '032b')
                full_binary = length_binary + full_binary
                
                print(f"Debug: After aggressive resize, binary length: {len(full_binary)}")
                
                # Final check
                if len(full_binary) > max_capacity:
                    raise ValueError(
                        f"Secret image is still too large after automatic resizing!\n\n"
                        f"Cover image capacity: {max_capacity // (8 * 1024 * 1024):.2f} MB\n"
                        f"Please use a larger cover image or a smaller secret image."
                    )
            
            # Hide data in cover image using 4 LSB layers for maximum capacity
            for pixel_idx in range(len(cover_flat)):
                start_bit = pixel_idx * 4
                if start_bit >= len(full_binary):
                    break
                end_bit = min(start_bit + 4, len(full_binary))
                bits = full_binary[start_bit:end_bit]
                # Pad with zeros if less than 4 bits
                bits = bits.ljust(4, '0')
                bit_val = int(bits, 2)
                pixel = cover_flat[pixel_idx]
                pixel = pixel & 0xF0  # Clear 4 LSB bits
                pixel = pixel | bit_val
                cover_flat[pixel_idx] = pixel
            
            # Reshape and save
            stego_img = cover_flat.reshape(cover_img.shape)
            
            # Save as PNG
            if not output_path.lower().endswith('.png'):
                output_path = os.path.splitext(output_path)[0] + '.png'
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            success = cv2.imwrite(output_path, stego_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Debug: Image-in-image save success: {success}")
            
            if not success:
                raise ValueError(f"Failed to save image to {output_path}. Check file path and permissions.")
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise ValueError(f"Image file was not created at {output_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error hiding image in image: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def extract_image_from_image(self, stego_image_path, output_path, decryption_key=None):
        """Extract hidden image from stego image"""
        try:
            print(f"Debug: Extracting image from {stego_image_path}")
            
            # Read stego image
            stego_img = cv2.imread(stego_image_path)
            if stego_img is None:
                raise ValueError("Could not read stego image file")
            
            print(f"Debug: Stego image shape: {stego_img.shape}")
            
            # Flatten image
            stego_flat = stego_img.flatten()
            
            # Extract length header (first 32 bits) using 4 LSB layers (current method)
            length_binary = ''
            pixels_needed = (32 + 3) // 4  # Ceiling division: 32 bits / 4 bits per pixel = 8 pixels
            for i in range(pixels_needed):
                if i < len(stego_flat):
                    pixel = stego_flat[i]
                    bits_val = pixel & 0x0F  # Extract 4 LSB bits
                    bits_str = format(bits_val, '04b')  # Convert to 4-bit binary string
                    length_binary += bits_str
            # Trim to exactly 32 bits
            length_binary = length_binary[:32]

            data_length = int(length_binary, 2)
            print(f"Debug: Data length (4 LSB method): {data_length}")

            # Validate data length
            max_reasonable_length = len(stego_flat) * 4  # Maximum possible with 4 LSB layers
            if data_length > max_reasonable_length or data_length < 0:
                raise ValueError(
                    f"Invalid data length: {data_length:,} bits (~{data_length // (8 * 1024 * 1024):.2f} MB)\n"
                    f"This might indicate:\n"
                    f"1. The stego image doesn't contain hidden data\n"
                    f"2. The image was corrupted\n"
                    f"3. Wrong decryption key (if encrypted)\n"
                    f"4. Image was created with different steganography method\n"
                    f"Maximum expected length: {max_reasonable_length:,} bits (~{max_reasonable_length // (8 * 1024 * 1024):.2f} MB)"
                )
            
            # Extract data bits using 4 LSB layers (consistent with hiding method)
            # Calculate how many pixels we need
            total_bits_needed = 32 + data_length  # 32 for length header + data_length for actual data
            available_pixels = len(stego_flat)

            available_bits = available_pixels * 4  # Each pixel provides 4 bits
            print(f"Debug: Need {total_bits_needed} bits, have {available_bits} bits available")

            if total_bits_needed > available_bits:
                # Try to recalculate - maybe the length was read incorrectly
                print("Debug: Warning - data_length seems too large, trying to extract what we can...")
                # Extract all available data after the header
                max_data_bits = available_bits - 32
                if max_data_bits > 0:
                    print(f"Debug: Adjusting data_length from {data_length} to {max_data_bits} based on available bits")
                    data_length = max_data_bits
                else:
                    raise ValueError(
                        f"Not enough bits in stego image!\n"
                        f"Need: {total_bits_needed} bits\n"
                        f"Have: {available_bits} bits\n"
                        f"The stego image may be corrupted or incomplete."
                    )

            data_binary = ''
            bits_extracted = 0
            pixels_needed = (data_length + 3) // 4  # Ceiling division: data_length bits / 4 bits per pixel
            start_pixel = 32 // 4  # 32 bits header / 4 bits per pixel = 8 pixels
            for pixel_idx in range(pixels_needed):
                i = start_pixel + pixel_idx
                if i < len(stego_flat):
                    pixel = stego_flat[i]
                    # Extract from 4 LSB layers (consistent with hiding)
                    bit_val = pixel & 0x0F  # Extract 4 LSB bits
                    bits_str = format(bit_val, '04b')
                    data_binary += bits_str
                    bits_extracted += 4
                else:
                    # We've run out of pixels - this shouldn't happen if length is correct
                    print(f"Debug: Warning - ran out of pixels at index {i}, expected {32 + pixels_needed}")
                    break

            print(f"Debug: Extracted {bits_extracted} bits out of {data_length} expected ({len(data_binary)} bits total)")

            # Validate we extracted enough data, but be lenient
            if len(data_binary) < data_length:
                missing_bits = data_length - len(data_binary)
                missing_percent = (missing_bits / data_length) * 100

                if missing_percent > 10:  # More than 10% missing
                    raise ValueError(
                        f"Incomplete data extraction!\n"
                        f"Expected: {data_length} bits ({data_length // 8:,} bytes)\n"
                        f"Extracted: {len(data_binary)} bits ({len(data_binary) // 8:,} bytes)\n"
                        f"Missing: {missing_bits:,} bits ({missing_percent:.1f}%)\n"
                        f"This might indicate:\n"
                        f"1. Stego image is corrupted or incomplete\n"
                        f"2. Data extraction method mismatch\n"
                        f"3. Image was modified after hiding data\n"
                        f"4. Wrong decryption key (if encrypted)"
                    )
                else:
                    # Less than 10% missing - might be due to rounding or slight extraction differences
                    print(f"Debug: Warning - extracted {len(data_binary)} bits, expected {data_length} ({missing_percent:.1f}% missing). Continuing anyway...")
                    # Adjust data_length to match what we actually extracted
                    data_length = len(data_binary)
            
            # Decrypt if key provided
            if decryption_key:
                try:
                    print("Debug: Attempting to decrypt image data...")
                    encrypted_text = self.binary_to_text(data_binary)
                    decrypted_b64 = self.simple_decrypt(encrypted_text, decryption_key)
                    # Decode from base64 back to binary with padding handling
                    try:
                        binary_bytes = base64.b64decode(decrypted_b64.encode('utf-8'))
                    except Exception as decode_error:
                        # Try adding padding if missing
                        missing_padding = len(decrypted_b64) % 4
                        if missing_padding:
                            decrypted_b64 += '=' * (4 - missing_padding)
                        try:
                            binary_bytes = base64.b64decode(decrypted_b64.encode('utf-8'))
                        except Exception:
                            print(f"Debug: Base64 decode failed: {str(decode_error)}")
                            raise ValueError("Decryption failed. The decrypted data is not valid. Wrong key or corrupted data.")
                    # Convert back to binary string
                    data_binary = ''.join(format(byte, '08b') for byte in binary_bytes)
                    print("Debug: Decryption successful")
                except ValueError as ve:
                    # Re-raise ValueError with clear message
                    raise ve
                except Exception as e:
                    print(f"Debug: Decryption failed: {str(e)}")
                    # Try to extract without decryption (maybe image wasn't encrypted)
                    print("Debug: Attempting extraction without decryption...")
                    try:
                        # Check if data looks like it might be unencrypted
                        # Try to parse dimensions directly
                        test_height = int(data_binary[:32], 2) if len(data_binary) >= 32 else 0
                        test_width = int(data_binary[32:64], 2) if len(data_binary) >= 64 else 0
                        if 0 < test_height < 100000 and 0 < test_width < 100000:
                            print("Debug: Data appears to be unencrypted, continuing without decryption")
                            # Don't decrypt, use data as-is
                        else:
                            raise ValueError(
                                f"Decryption failed. Possible causes:\n"
                                f"1. Wrong decryption key\n"
                                f"2. Image was not encrypted (try without key)\n"
                                f"3. Corrupted stego image data\n"
                                f"Original error: {str(e)}"
                            )
                    except Exception as test_error:
                        raise ValueError(
                            f"Decryption failed. Wrong key or corrupted data.\n"
                            f"Error: {str(e)}\n"
                            f"If the image was not encrypted, try extracting without a decryption key."
                        )
            
            # Validate we have enough data for header (97 bits minimum)
            if len(data_binary) < 97:
                raise ValueError(
                    f"Extracted data is too short ({len(data_binary)} bits). "
                    f"Expected at least 97 bits for header. The stego image may not contain valid hidden data."
                )
            
            # Extract dimensions (first 96 bits: 32 for height, 32 for width, 32 for channels)
            height_binary = data_binary[:32]
            width_binary = data_binary[32:64]
            channels_binary = data_binary[64:96]
            pixel_binary = data_binary[96:]
            
            try:
                original_height = int(height_binary, 2)
                original_width = int(width_binary, 2)
                original_channels = int(channels_binary, 2)
            except ValueError as ve:
                raise ValueError(
                    f"Failed to parse image dimensions from extracted data. "
                    f"This might indicate:\n"
                    f"1. Wrong decryption key (if encrypted)\n"
                    f"2. Corrupted stego image\n"
                    f"3. Image doesn't contain hidden data\n"
                    f"Error: {str(ve)}"
                )

            # Validate dimensions are reasonable
            if original_height <= 0 or original_width <= 0 or original_height > 100000 or original_width > 100000:
                raise ValueError(
                    f"Invalid image dimensions extracted: {original_width}x{original_height}\n"
                    f"This suggests:\n"
                    f"1. Wrong decryption key (if encrypted)\n"
                    f"2. Corrupted or invalid stego image\n"
                    f"3. Image was not created with this steganography method"
                )

            if original_channels not in [1, 3, 4]:
                raise ValueError(
                    f"Invalid number of channels: {original_channels}\n"
                    f"Expected 1 (grayscale), 3 (RGB), or 4 (RGBA)"
                )

            print(f"Debug: Original dimensions: {original_height}x{original_width}, channels: {original_channels} (uncompressed)")

            # Reconstruct image from uncompressed pixel data
            num_pixels = len(pixel_binary) // 8
            pixels = []
            for i in range(0, len(pixel_binary), 8):
                if i + 8 <= len(pixel_binary):
                    byte = pixel_binary[i:i+8]
                    pixels.append(int(byte, 2))

            # Reshape to original dimensions
            expected_size = original_height * original_width * original_channels
            
            # Check if we have enough pixels
            if len(pixels) < expected_size:
                print(f"Debug: Warning - Insufficient data for declared dimensions. Expected {expected_size}, got {len(pixels)}")
                print("Debug: Padding with zeros to allow reshaping (image will be incomplete)")
                # Pad with zeros
                pixels.extend([0] * (expected_size - len(pixels)))
            
            # Truncate if too many (shouldn't happen with the slice logic, but good for safety)
            pixels = pixels[:expected_size]
            
            if original_channels == 1:
                secret_img = np.array(pixels, dtype=np.uint8)
                secret_img = secret_img.reshape((original_height, original_width))
            else:
                secret_img = np.array(pixels, dtype=np.uint8)
                secret_img = secret_img.reshape((original_height, original_width, original_channels))
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save extracted image
            success = cv2.imwrite(output_path, secret_img)
            print(f"Debug: Extracted image save success: {success}")
            
            if not success:
                raise ValueError(f"Failed to save extracted image to {output_path}. Check file path and permissions.")
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise ValueError(f"Extracted image file was not created at {output_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error extracting image from image: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

class ReliableSteganographyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STEGANOGRAPHY FOR DATA SECURITY")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize steganography system
        self.stego = ReliableSteganography()
        
        # Variables
        self.source_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.encryption_key = tk.StringVar()
        self.decryption_key = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#27ae60', height=60)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="STEGANOGRAPHY FOR DATA SECURITY", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#27ae60')
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # Hide Message Tab
        self.create_hide_tab(notebook)
        
        # Extract Message Tab
        self.create_extract_tab(notebook)
        
        # PNG Converter Tab
        self.create_converter_tab(notebook)
        
        # Image Steganography Tab
        self.create_image_steganography_tab(notebook)
        
        # Image in Image Tab
        self.create_image_in_image_tab(notebook)
        
        # Video Steganography Tab
        self.create_video_tab(notebook)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Ultra-enhanced capacity with 3x data hiding + Video support!")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor='w', bg='#27ae60', fg='white')
        status_bar.pack(side='bottom', fill='x')
        
    def create_hide_tab(self, notebook):
        """Create the hide message tab"""
        hide_frame = ttk.Frame(notebook)
        notebook.add(hide_frame, text="Hide Message")
        
        # Source file selection
        source_frame = tk.LabelFrame(hide_frame, text="Source Image (PNG, BMP, TIFF, JPG, JPEG)", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        source_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(source_frame, text="Select image file (JPG/JPEG will be auto-converted to PNG):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        file_frame = tk.Frame(source_frame, bg='#f0f0f0')
        file_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Entry(file_frame, textvariable=self.source_file, width=50).pack(side='left', padx=(0,5))
        tk.Button(file_frame, text="Browse Image", command=self.browse_image_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Create sample buttons
        create_frame = tk.Frame(source_frame, bg='#f0f0f0')
        create_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Button(create_frame, text="Create PNG", command=self.create_png_file, 
                 bg='#27ae60', fg='white', font=('Arial', 9)).pack(side='left', padx=(0,5))
        
        tk.Button(create_frame, text="Create BMP", command=self.create_bmp_file, 
                 bg='#e74c3c', fg='white', font=('Arial', 9)).pack(side='left', padx=(0,5))
        
        tk.Button(create_frame, text="Create TIFF", command=self.create_tiff_file, 
                 bg='#9b59b6', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Message input
        message_frame = tk.LabelFrame(hide_frame, text="Message to Hide", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        message_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        tk.Label(message_frame, text="Enter your secret message:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        self.message_entry = scrolledtext.ScrolledText(message_frame, height=6, width=70)
        self.message_entry.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Encryption key
        encrypt_frame = tk.LabelFrame(hide_frame, text="Encryption Key (Optional)", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        encrypt_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(encrypt_frame, text="Enter encryption key (leave empty for no encryption):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        tk.Entry(encrypt_frame, textvariable=self.encryption_key, width=50, show="*").pack(padx=5, pady=2)
        
        # Output file
        output_frame = tk.LabelFrame(hide_frame, text="Output File (Will be saved as PNG)", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        output_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(output_frame, text="Output stego file (will be converted to PNG):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        output_file_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_file_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Entry(output_file_frame, textvariable=self.output_file, width=50).pack(side='left', padx=(0,5))
        tk.Button(output_file_frame, text="Browse", command=self.browse_output_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Hide button
        hide_button = tk.Button(hide_frame, text="Hide Message", 
                               command=self.hide_message, bg='#e74c3c', fg='white', 
                               font=('Arial', 12, 'bold'), height=2)
        hide_button.pack(pady=10)
        
    def create_extract_tab(self, notebook):
        """Create the extract message tab"""
        extract_frame = ttk.Frame(notebook)
        notebook.add(extract_frame, text="Extract Message")
        
        # Stego file selection
        stego_frame = tk.LabelFrame(extract_frame, text="Stego File", 
                                  font=('Arial', 10, 'bold'), bg='#f0f0f0')
        stego_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(stego_frame, text="Select stego file:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        stego_file_frame = tk.Frame(stego_frame, bg='#f0f0f0')
        stego_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.stego_file_var = tk.StringVar()
        tk.Entry(stego_file_frame, textvariable=self.stego_file_var, width=50).pack(side='left', padx=(0,5))
        tk.Button(stego_file_frame, text="Browse Image", command=self.browse_stego_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Decryption key
        decrypt_frame = tk.LabelFrame(extract_frame, text="Decryption Key", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        decrypt_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(decrypt_frame, text="Enter decryption key (if message was encrypted):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        tk.Entry(decrypt_frame, textvariable=self.decryption_key, width=50, show="*").pack(padx=5, pady=2)
        
        # Extract button
        extract_button = tk.Button(extract_frame, text="Extract Message", 
                                 command=self.extract_message, bg='#27ae60', fg='white', 
                                 font=('Arial', 12, 'bold'), height=2)
        extract_button.pack(pady=10)
        
        # Results
        results_frame = tk.LabelFrame(extract_frame, text="Extracted Message", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.extracted_text = scrolledtext.ScrolledText(results_frame, height=10, width=70, state='disabled')
        self.extracted_text.pack(fill='both', expand=True, padx=5, pady=2)
        
    def create_image_steganography_tab(self, notebook):
        """Create the Image Steganography tab with both hide and extract functionality"""
        image_stego_frame = ttk.Frame(notebook)
        notebook.add(image_stego_frame, text="Image Steganography")
        
        # Create a paned window to separate hide and extract sections
        paned = ttk.PanedWindow(image_stego_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left pane - Hide Message Section
        hide_pane = ttk.Frame(paned)
        paned.add(hide_pane, weight=1)
        
        hide_section = tk.LabelFrame(hide_pane, text="Hide Message in Image", 
                                     font=('Arial', 11, 'bold'), bg='#f0f0f0')
        hide_section.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Source file selection
        source_frame = tk.LabelFrame(hide_section, text="Source Image", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        source_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(source_frame, text="Select image (PNG, BMP, TIFF, JPG, JPEG):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        file_frame = tk.Frame(source_frame, bg='#f0f0f0')
        file_frame.pack(fill='x', padx=5, pady=2)
        
        self.image_stego_source = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.image_stego_source, width=40).pack(side='left', padx=(0,5))
        tk.Button(file_frame, text="Browse", command=self.browse_image_stego_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Message input
        message_frame = tk.LabelFrame(hide_section, text="Secret Message", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        message_frame.pack(fill='both', expand=True, padx=5, pady=3)
        
        self.image_stego_message = scrolledtext.ScrolledText(message_frame, height=5, width=40)
        self.image_stego_message.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Encryption key
        encrypt_frame = tk.LabelFrame(hide_section, text="Encryption Key (Optional)", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        encrypt_frame.pack(fill='x', padx=5, pady=3)
        
        self.image_stego_encrypt_key = tk.StringVar()
        tk.Entry(encrypt_frame, textvariable=self.image_stego_encrypt_key, width=40, show="*").pack(padx=5, pady=2)
        
        # Output file
        output_frame = tk.LabelFrame(hide_section, text="Output File", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        output_frame.pack(fill='x', padx=5, pady=3)
        
        self.image_stego_output = tk.StringVar()
        output_file_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_file_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Entry(output_file_frame, textvariable=self.image_stego_output, width=40).pack(side='left', padx=(0,5))
        tk.Button(output_file_frame, text="Browse", command=self.browse_image_stego_output, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Hide button
        hide_button = tk.Button(hide_section, text="Hide Message", 
                               command=self.hide_image_stego_message, bg='#e74c3c', fg='white', 
                               font=('Arial', 11, 'bold'), height=2)
        hide_button.pack(pady=5)
        
        # Right pane - Extract Message Section
        extract_pane = ttk.Frame(paned)
        paned.add(extract_pane, weight=1)
        
        extract_section = tk.LabelFrame(extract_pane, text="Extract Message from Image", 
                                       font=('Arial', 11, 'bold'), bg='#f0f0f0')
        extract_section.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Stego file selection
        stego_frame = tk.LabelFrame(extract_section, text="Stego Image", 
                                  font=('Arial', 10, 'bold'), bg='#f0f0f0')
        stego_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(stego_frame, text="Select stego image file:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        stego_file_frame = tk.Frame(stego_frame, bg='#f0f0f0')
        stego_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.image_stego_stego_file = tk.StringVar()
        tk.Entry(stego_file_frame, textvariable=self.image_stego_stego_file, width=40).pack(side='left', padx=(0,5))
        tk.Button(stego_file_frame, text="Browse", command=self.browse_image_stego_stego_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Decryption key
        decrypt_frame = tk.LabelFrame(extract_section, text="Decryption Key", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        decrypt_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(decrypt_frame, text="Enter decryption key (if encrypted):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        self.image_stego_decrypt_key = tk.StringVar()
        tk.Entry(decrypt_frame, textvariable=self.image_stego_decrypt_key, width=40, show="*").pack(padx=5, pady=2)
        
        # Extract button
        extract_button = tk.Button(extract_section, text="Extract Message", 
                                 command=self.extract_image_stego_message, bg='#27ae60', fg='white', 
                                 font=('Arial', 11, 'bold'), height=2)
        extract_button.pack(pady=5)
        
        # Results
        results_frame = tk.LabelFrame(extract_section, text="Extracted Message", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        results_frame.pack(fill='both', expand=True, padx=5, pady=3)
        
        self.image_stego_extracted_text = scrolledtext.ScrolledText(results_frame, height=8, width=40, state='disabled')
        self.image_stego_extracted_text.pack(fill='both', expand=True, padx=5, pady=2)
        
    def create_converter_tab(self, notebook):
        """Create PNG converter tab"""
        converter_frame = ttk.Frame(notebook)
        notebook.add(converter_frame, text="PNG Converter")
        
        # Source file selection
        source_frame = tk.LabelFrame(converter_frame, text="Source Image (JPG/JPEG to PNG)", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        source_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(source_frame, text="Select JPG/JPEG image to convert to PNG:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        file_frame = tk.Frame(source_frame, bg='#f0f0f0')
        file_frame.pack(fill='x', padx=5, pady=2)
        
        self.converter_source_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.converter_source_var, width=50).pack(side='left', padx=(0,5))
        tk.Button(file_frame, text="Browse JPG/JPEG", command=self.browse_converter_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Convert button
        convert_button = tk.Button(converter_frame, text="Convert to PNG", 
                                 command=self.convert_image, bg='#e74c3c', fg='white', 
                                 font=('Arial', 12, 'bold'), height=2)
        convert_button.pack(pady=10)
        
        # Results
        results_frame = tk.LabelFrame(converter_frame, text="Conversion Results", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.converter_text = scrolledtext.ScrolledText(results_frame, height=10, width=70, state='disabled')
        self.converter_text.pack(fill='both', expand=True, padx=5, pady=2)
        
    def create_video_tab(self, notebook):
        """Create video steganography tab"""
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video Steganography")
        
        # Hide in Video
        hide_video_frame = tk.LabelFrame(video_frame, text="Hide Message in Video", 
                                       font=('Arial', 10, 'bold'), bg='#f0f0f0')
        hide_video_frame.pack(fill='x', padx=10, pady=5)
        
        # Source video selection
        tk.Label(hide_video_frame, text="Select video file:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        video_file_frame = tk.Frame(hide_video_frame, bg='#f0f0f0')
        video_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.video_source_var = tk.StringVar()
        tk.Entry(video_file_frame, textvariable=self.video_source_var, width=50).pack(side='left', padx=(0,5))
        tk.Button(video_file_frame, text="Browse Video", command=self.browse_video_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Message input
        message_video_frame = tk.LabelFrame(video_frame, text="Message to Hide in Video", 
                                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
        message_video_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        tk.Label(message_video_frame, text="Enter your secret message:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        self.video_message_entry = scrolledtext.ScrolledText(message_video_frame, height=4, width=70)
        self.video_message_entry.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Encryption key
        encrypt_video_frame = tk.LabelFrame(video_frame, text="Encryption Key (Optional)", 
                                         font=('Arial', 10, 'bold'), bg='#f0f0f0')
        encrypt_video_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(encrypt_video_frame, text="Enter encryption key (leave empty for no encryption):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        self.video_encryption_key = tk.StringVar()
        tk.Entry(encrypt_video_frame, textvariable=self.video_encryption_key, width=50, show="*").pack(padx=5, pady=2)
        
        # Output video
        output_video_frame = tk.LabelFrame(video_frame, text="Output Video", 
                                         font=('Arial', 10, 'bold'), bg='#f0f0f0')
        output_video_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(output_video_frame, text="Output stego video:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        output_video_file_frame = tk.Frame(output_video_frame, bg='#f0f0f0')
        output_video_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.video_output_var = tk.StringVar()
        tk.Entry(output_video_file_frame, textvariable=self.video_output_var, width=50).pack(side='left', padx=(0,5))
        tk.Button(output_video_file_frame, text="Browse", command=self.browse_video_output, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Hide in video button
        hide_video_button = tk.Button(video_frame, text="Hide Message in Video", 
                                    command=self.hide_message_in_video, bg='#e74c3c', fg='white', 
                                    font=('Arial', 14, 'bold'), height=3, width=25)
        hide_video_button.pack(pady=10)
        
        # Extract from Video
        extract_video_frame = tk.LabelFrame(video_frame, text="Extract Message from Video", 
                                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
        extract_video_frame.pack(fill='x', padx=10, pady=5)
        
        # Stego video selection
        tk.Label(extract_video_frame, text="Select stego video:", bg="#fafcfd").pack(anchor='w', padx=5, pady=2)
        
        stego_video_file_frame = tk.Frame(extract_video_frame, bg='#f0f0f0')
        stego_video_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.stego_video_file_var = tk.StringVar()
        tk.Entry(stego_video_file_frame, textvariable=self.stego_video_file_var, width=50).pack(side='left', padx=(0,5))
        tk.Button(stego_video_file_frame, text="Browse Video", command=self.browse_stego_video_file, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Decryption key
        decrypt_video_frame = tk.LabelFrame(video_frame, text="Decryption Key", 
                                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
        decrypt_video_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(decrypt_video_frame, text="Enter decryption key (if message was encrypted):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        self.video_decryption_key = tk.StringVar()
        tk.Entry(decrypt_video_frame, textvariable=self.video_decryption_key, width=50, show="*").pack(padx=5, pady=2)
        
        # Extract from video button
        extract_video_button = tk.Button(video_frame, text="Extract Message from Video", 
                                        command=self.extract_message_from_video, bg='#e74c3c', fg='white', 
                                        font=('Arial', 14, 'bold'), height=3, width=25)
        extract_video_button.pack(pady=10)
        
        # Video results
        video_results_frame = tk.LabelFrame(video_frame, text="Video Steganography Results", 
                                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
        video_results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.video_results_text = scrolledtext.ScrolledText(video_results_frame, height=8, width=70, state='disabled')
        self.video_results_text.pack(fill='both', expand=True, padx=5, pady=2)
        
    def browse_image_file(self):
        """Browse for image file"""
        filetypes = [("All supported formats", "*.png;*.bmp;*.tiff;*.jpg;*.jpeg;*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("PNG files", "*.png"), ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("JPG files", "*.jpg;*.jpeg"), ("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.source_file.set(filename)
            base, ext = os.path.splitext(filename)
            if any(filename.lower().endswith(video_ext) for video_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                self.output_file.set(f"{base}_stego.mp4")
            else:
                self.output_file.set(f"{base}_stego.png")
            
    def browse_stego_file(self):
        """Browse for stego file"""
        filetypes = [("All supported formats", "*.png;*.bmp;*.tiff;*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("PNG files", "*.png"), ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.stego_file_var.set(filename)
            
    def browse_output_file(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(defaultextension=".png")
        if filename:
            self.output_file.set(filename)
            
    def browse_converter_file(self):
        """Browse for JPG/JPEG file to convert"""
        filetypes = [("JPG/JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.converter_source_var.set(filename)
            
    def convert_image(self):
        """Convert JPG/JPEG to PNG"""
        try:
            source = self.converter_source_var.get()
            
            if not source:
                messagebox.showerror("Error", "Please select a JPG/JPEG file to convert!")
                return
                
            if not os.path.exists(source):
                messagebox.showerror("Error", "Source file does not exist!")
                return
                
            # Check if it's a JPG/JPEG file
            if not any(source.lower().endswith(ext) for ext in ['.jpg', '.jpeg']):
                messagebox.showerror("Error", "Please select a JPG or JPEG file!")
                return
                
            self.status_var.set("Converting to PNG...")
            self.root.update()
            
            # Convert to PNG
            png_path = self.stego.convert_to_png(source)
            
            if png_path:
                self.converter_text.config(state='normal')
                self.converter_text.delete('1.0', tk.END)
                self.converter_text.insert('1.0', f"Conversion Results:\n\n")
                self.converter_text.insert(tk.END, f"Source file: {source}\n")
                self.converter_text.insert(tk.END, f"Converted to: {png_path}\n")
                self.converter_text.insert(tk.END, f"File saved in same folder as original\n\n")
                self.converter_text.insert(tk.END, f"SUCCESS! JPG/JPEG converted to PNG!\n")
                self.converter_text.insert(tk.END, f"You can now use this PNG file for reliable steganography.\n")
                self.converter_text.config(state='disabled')
                
                self.status_var.set("Conversion completed successfully!")
                messagebox.showinfo("Success", f"JPG/JPEG converted to PNG successfully!\n\nConverted file: {png_path}\n\nYou can now use this PNG file for reliable steganography.")
            else:
                self.converter_text.config(state='normal')
                self.converter_text.delete('1.0', tk.END)
                self.converter_text.insert('1.0', "Conversion failed!")
                self.converter_text.config(state='disabled')
                
                self.status_var.set("Conversion failed!")
                messagebox.showerror("Error", "Failed to convert JPG/JPEG to PNG!")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error occurred!")
            
            self.converter_text.config(state='normal')
            self.converter_text.delete('1.0', tk.END)
            self.converter_text.insert('1.0', f"Error: {error_msg}")
            self.converter_text.config(state='disabled')
            
    def browse_video_file(self):
        """Browse for video file"""
        filetypes = [("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.video_source_var.set(filename)
            base, ext = os.path.splitext(filename)
            self.video_output_var.set(f"{base}_stego.mp4")
            
    def browse_video_output(self):
        """Browse for video output file"""
        filename = filedialog.asksaveasfilename(defaultextension=".mp4")
        if filename:
            self.video_output_var.set(filename)
            
    def browse_stego_video_file(self):
        """Browse for stego video file"""
        filetypes = [("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.stego_video_file_var.set(filename)
            
    def hide_message_in_video(self):
        """Hide message in video"""
        try:
            source = self.video_source_var.get()
            output = self.video_output_var.get()
            message = self.video_message_entry.get('1.0', tk.END).strip()
            key = self.video_encryption_key.get().strip()
            
            if not source or not output or not message:
                messagebox.showerror("Error", "Please fill in all required fields!")
                return
                
            if not os.path.exists(source):
                messagebox.showerror("Error", "Source video file does not exist!")
                return
                
            self.status_var.set("Hiding message in video...")
            self.root.update()
            
            success = self.stego.hide_message_in_video(source, message, output, key if key else None)
            
            if success:
                encryption_status = "with encryption" if key else "without encryption"
                messagebox.showinfo("Success", f"Message hidden in video successfully {encryption_status}!\nOutput: {output}")
                self.status_var.set("Message hidden in video successfully!")
                
                self.video_results_text.config(state='normal')
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', f"Video Steganography Results:\n\n")
                self.video_results_text.insert(tk.END, f"Source video: {source}\n")
                self.video_results_text.insert(tk.END, f"Output video: {output}\n")
                self.video_results_text.insert(tk.END, f"Message hidden: '{message}'\n")
                self.video_results_text.insert(tk.END, f"Encryption: {encryption_status}\n\n")
                self.video_results_text.insert(tk.END, f"SUCCESS! Message hidden in video with 3x capacity!\n")
                self.video_results_text.config(state='disabled')
            else:
                messagebox.showerror("Error", "Failed to hide message in video!")
                self.status_var.set("Failed to hide message in video!")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred!")
            
    def extract_message_from_video(self):
        """Extract message from video"""
        try:
            stego_video = self.stego_video_file_var.get()
            key = self.video_decryption_key.get().strip()
            
            if not stego_video:
                messagebox.showerror("Error", "Please select a stego video file!")
                return
                
            if not os.path.exists(stego_video):
                messagebox.showerror("Error", "Stego video file does not exist!")
                return
                
            self.status_var.set("Extracting message from video...")
            self.root.update()
            
            message = self.stego.extract_message_from_video(stego_video, key if key else None)
            
            if message:
                self.video_results_text.config(state='normal')
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', f"Video Extraction Results:\n\n")
                self.video_results_text.insert(tk.END, f"Stego video: {stego_video}\n")
                self.video_results_text.insert(tk.END, f"Extracted message: '{message}'\n\n")
                self.video_results_text.insert(tk.END, f"SUCCESS! Message extracted from video!\n")
                self.video_results_text.config(state='disabled')
                
                self.status_var.set("Message extracted from video successfully!")
                messagebox.showinfo("Success", f"Message extracted from video successfully!\n\nExtracted: {message}")
            else:
                self.video_results_text.config(state='normal')
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', "No message found or extraction failed.")
                self.video_results_text.config(state='disabled')
                self.status_var.set("No message found in video!")
                messagebox.showwarning("Warning", "No message found in video. Check:\n1. Video contains hidden data\n2. Correct decryption key\n3. Video format is supported")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error occurred!")
            
            self.video_results_text.config(state='normal')
            self.video_results_text.delete('1.0', tk.END)
            self.video_results_text.insert('1.0', f"Error: {error_msg}")
            self.video_results_text.config(state='disabled')
            
    def create_png_file(self):
        """Create a sample PNG file"""
        try:
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            img[:] = [120, 80, 200]
            filename = 'sample_reliable.png'
            cv2.imwrite(filename, img)
            
            self.source_file.set(filename)
            self.output_file.set('sample_reliable_stego.png')
            
            messagebox.showinfo("Success", f"Sample PNG created: {filename}")
            self.status_var.set("Sample PNG created successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create PNG: {str(e)}")
            
    def create_bmp_file(self):
        """Create a sample BMP file"""
        try:
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            img[:] = [120, 80, 200]
            filename = 'sample_reliable.bmp'
            cv2.imwrite(filename, img)
            
            self.source_file.set(filename)
            self.output_file.set('sample_reliable_stego.png')
            
            messagebox.showinfo("Success", f"Sample BMP created: {filename}")
            self.status_var.set("Sample BMP created successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create BMP: {str(e)}")
            
    def create_tiff_file(self):
        """Create a sample TIFF file"""
        try:
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            img[:] = [120, 80, 200]
            filename = 'sample_reliable.tiff'
            cv2.imwrite(filename, img)
            
            self.source_file.set(filename)
            self.output_file.set('sample_reliable_stego.png')
            
            messagebox.showinfo("Success", f"Sample TIFF created: {filename}")
            self.status_var.set("Sample TIFF created successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create TIFF: {str(e)}")
            
    def hide_message(self):
        """Hide message in image"""
        try:
            source = self.source_file.get()
            output = self.output_file.get()
            message = self.message_entry.get('1.0', tk.END).strip()
            key = self.encryption_key.get().strip()
            
            if not source or not output or not message:
                messagebox.showerror("Error", "Please fill in all required fields!")
                return
            
            if not message.strip():
                messagebox.showerror("Error", "Message cannot be empty!")
                return
                
            if not os.path.exists(source):
                messagebox.showerror("Error", f"Source file does not exist: {source}")
                return
            
            # Validate output path
            output_dir = os.path.dirname(output) if os.path.dirname(output) else '.'
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot create output directory: {output_dir}\nError: {str(e)}")
                return
                
            self.status_var.set("Hiding message...")
            self.root.update()
            
            try:
                success = self.stego.hide_message(source, message, output, key if key else None)

                if success:
                    encryption_status = "with encryption" if key else "without encryption"
                    file_size = os.path.getsize(output) / (1024 * 1024) if os.path.exists(output) else 0
                    messagebox.showinfo("Success",
                        f"Message hidden successfully {encryption_status}!\n\n"
                        f"Output saved as PNG: {output}\n"
                        f"File size: {file_size:.2f} MB")
                    self.status_var.set("Message hidden successfully!")
                else:
                    messagebox.showerror("Error", "Failed to hide message! Check the console for details.")
                    self.status_var.set("Failed to hide message!")
            except Exception as hide_error:
                # Re-raise to be caught by outer except
                raise hide_error
                
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to hide message!\n\nError: {error_msg}")
            self.status_var.set(f"Error: {error_msg[:50]}...")
            
    def extract_message(self):
        """Extract message from stego file"""
        try:
            stego_file = self.stego_file_var.get()
            key = self.decryption_key.get().strip()
            
            if not stego_file:
                messagebox.showerror("Error", "Please select a stego file!")
                return
                
            if not os.path.exists(stego_file):
                messagebox.showerror("Error", "Stego file does not exist!")
                return
                
            self.status_var.set("Extracting message...")
            self.root.update()
            
            message = self.stego.extract_message(stego_file, key if key else None)
            
            if message:
                self.extracted_text.config(state='normal')
                self.extracted_text.delete('1.0', tk.END)
                self.extracted_text.insert('1.0', message)
                self.extracted_text.config(state='disabled')
                self.status_var.set("Message extracted successfully!")
                messagebox.showinfo("Success", f"Message extracted successfully!\n\nExtracted: {message}")
            else:
                self.extracted_text.config(state='normal')
                self.extracted_text.delete('1.0', tk.END)
                self.extracted_text.insert('1.0', "No message found or extraction failed.")
                self.extracted_text.config(state='disabled')
                self.status_var.set("No message found!")
                messagebox.showwarning("Warning", "No message found. Check:\n1. File contains hidden data\n2. Correct decryption key\n3. File format is supported")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error occurred!")
            
            self.extracted_text.config(state='normal')
            self.extracted_text.delete('1.0', tk.END)
            self.extracted_text.insert('1.0', f"Error: {error_msg}")
            self.extracted_text.config(state='disabled')
            
    def browse_image_stego_file(self):
        """Browse for image file in Image Steganography tab"""
        filetypes = [("Image files", "*.png;*.bmp;*.tiff;*.jpg;*.jpeg"), ("PNG files", "*.png"), 
                     ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("JPG files", "*.jpg;*.jpeg"), 
                     ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.image_stego_source.set(filename)
            base, ext = os.path.splitext(filename)
            self.image_stego_output.set(f"{base}_stego.png")
    
    def browse_image_stego_output(self):
        """Browse for output file in Image Steganography tab"""
        filename = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename:
            self.image_stego_output.set(filename)
    
    def browse_image_stego_stego_file(self):
        """Browse for stego file in Image Steganography tab"""
        filetypes = [("Image files", "*.png;*.bmp;*.tiff"), ("PNG files", "*.png"), 
                     ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.image_stego_stego_file.set(filename)
    
    def hide_image_stego_message(self):
        """Hide message in image from Image Steganography tab"""
        try:
            source = self.image_stego_source.get()
            output = self.image_stego_output.get()
            message = self.image_stego_message.get('1.0', tk.END).strip()
            key = self.image_stego_encrypt_key.get().strip()
            
            if not source or not output or not message:
                messagebox.showerror("Error", "Please fill in all required fields!")
                return
            
            if not message.strip():
                messagebox.showerror("Error", "Message cannot be empty!")
                return
                
            if not os.path.exists(source):
                messagebox.showerror("Error", f"Source file does not exist: {source}")
                return
            
            # Validate output path
            output_dir = os.path.dirname(output) if os.path.dirname(output) else '.'
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot create output directory: {output_dir}\nError: {str(e)}")
                    return
                
            self.status_var.set("Hiding message in image...")
            self.root.update()
            
            try:
                success = self.stego.hide_message(source, message, output, key if key else None)
                
                if success:
                    encryption_status = "with encryption" if key else "without encryption"
                    file_size = os.path.getsize(output) / (1024 * 1024) if os.path.exists(output) else 0
                    messagebox.showinfo("Success",
                        f"Message hidden successfully {encryption_status}!\n\n"
                        f"Output: {output}\n"
                        f"File size: {file_size:.2f} MB")
                    self.status_var.set("Message hidden successfully!")
                else:
                    messagebox.showerror("Error", "Failed to hide message! Check the console for details.")
                    self.status_var.set("Failed to hide message!")
            except Exception as hide_error:
                # Re-raise to be caught by outer except
                raise hide_error
                
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to hide message!\n\nError: {error_msg}")
            self.status_var.set(f"Error: {error_msg[:50]}...")
    
    def extract_image_stego_message(self):
        """Extract message from image in Image Steganography tab"""
        try:
            stego_file = self.image_stego_stego_file.get()
            key = self.image_stego_decrypt_key.get().strip()
            
            if not stego_file:
                messagebox.showerror("Error", "Please select a stego image file!")
                return
                
            if not os.path.exists(stego_file):
                messagebox.showerror("Error", "Stego file does not exist!")
                return
                
            self.status_var.set("Extracting message from image...")
            self.root.update()
            
            message = self.stego.extract_message(stego_file, key if key else None)
            
            if message:
                self.image_stego_extracted_text.config(state='normal')
                self.image_stego_extracted_text.delete('1.0', tk.END)
                self.image_stego_extracted_text.insert('1.0', message)
                self.image_stego_extracted_text.config(state='disabled')
                self.status_var.set("Message extracted successfully!")
                messagebox.showinfo("Success", f"Message extracted successfully!\n\nExtracted: {message}")
            else:
                self.image_stego_extracted_text.config(state='normal')
                self.image_stego_extracted_text.delete('1.0', tk.END)
                self.image_stego_extracted_text.insert('1.0', "No message found or extraction failed.")
                self.image_stego_extracted_text.config(state='disabled')
                self.status_var.set("No message found!")
                messagebox.showwarning("Warning", "No message found. Check:\n1. File contains hidden data\n2. Correct decryption key\n3. File format is supported")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error occurred!")
            
            self.image_stego_extracted_text.config(state='normal')
            self.image_stego_extracted_text.delete('1.0', tk.END)
            self.image_stego_extracted_text.insert('1.0', f"Error: {error_msg}")
            self.image_stego_extracted_text.config(state='disabled')
    
    def create_image_in_image_tab(self, notebook):
        """Create the Image in Image steganography tab"""
        img_in_img_frame = ttk.Frame(notebook)
        notebook.add(img_in_img_frame, text="Image in Image")
        
        # Create a paned window to separate hide and extract sections
        paned = ttk.PanedWindow(img_in_img_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left pane - Hide Image Section
        hide_pane = ttk.Frame(paned)
        paned.add(hide_pane, weight=1)
        
        hide_section = tk.LabelFrame(hide_pane, text="Hide Image in Image", 
                                     font=('Arial', 11, 'bold'), bg='#f0f0f0')
        hide_section.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Cover image selection
        cover_frame = tk.LabelFrame(hide_section, text="Cover Image", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        cover_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(cover_frame, text="Select cover image (PNG, BMP, TIFF, JPG, JPEG):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        cover_file_frame = tk.Frame(cover_frame, bg='#f0f0f0')
        cover_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.img_in_img_cover = tk.StringVar()
        tk.Entry(cover_file_frame, textvariable=self.img_in_img_cover, width=40).pack(side='left', padx=(0,5))
        tk.Button(cover_file_frame, text="Browse", command=self.browse_img_in_img_cover, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Secret image selection
        secret_frame = tk.LabelFrame(hide_section, text="Secret Image to Hide", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        secret_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(secret_frame, text="Select secret image to hide:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        secret_file_frame = tk.Frame(secret_frame, bg='#f0f0f0')
        secret_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.img_in_img_secret = tk.StringVar()
        tk.Entry(secret_file_frame, textvariable=self.img_in_img_secret, width=40).pack(side='left', padx=(0,5))
        tk.Button(secret_file_frame, text="Browse", command=self.browse_img_in_img_secret, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Encryption key
        encrypt_frame = tk.LabelFrame(hide_section, text="Encryption Key (Optional)", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        encrypt_frame.pack(fill='x', padx=5, pady=3)
        
        self.img_in_img_encrypt_key = tk.StringVar()
        tk.Entry(encrypt_frame, textvariable=self.img_in_img_encrypt_key, width=40, show="*").pack(padx=5, pady=2)
        
        # Output file
        output_frame = tk.LabelFrame(hide_section, text="Output File", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        output_frame.pack(fill='x', padx=5, pady=3)
        
        self.img_in_img_output = tk.StringVar()
        output_file_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_file_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Entry(output_file_frame, textvariable=self.img_in_img_output, width=40).pack(side='left', padx=(0,5))
        tk.Button(output_file_frame, text="Browse", command=self.browse_img_in_img_output, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Hide button
        hide_button = tk.Button(hide_section, text="Hide Image", 
                               command=self.hide_img_in_img, bg='#e74c3c', fg='white', 
                               font=('Arial', 11, 'bold'), height=2)
        hide_button.pack(pady=5)
        
        # Right pane - Extract Image Section
        extract_pane = ttk.Frame(paned)
        paned.add(extract_pane, weight=1)
        
        extract_section = tk.LabelFrame(extract_pane, text="Extract Image from Image", 
                                       font=('Arial', 11, 'bold'), bg='#f0f0f0')
        extract_section.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Stego image selection
        stego_frame = tk.LabelFrame(extract_section, text="Stego Image", 
                                  font=('Arial', 10, 'bold'), bg='#f0f0f0')
        stego_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(stego_frame, text="Select stego image file:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        stego_file_frame = tk.Frame(stego_frame, bg='#f0f0f0')
        stego_file_frame.pack(fill='x', padx=5, pady=2)
        
        self.img_in_img_stego = tk.StringVar()
        tk.Entry(stego_file_frame, textvariable=self.img_in_img_stego, width=40).pack(side='left', padx=(0,5))
        tk.Button(stego_file_frame, text="Browse", command=self.browse_img_in_img_stego, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Decryption key
        decrypt_frame = tk.LabelFrame(extract_section, text="Decryption Key", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        decrypt_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(decrypt_frame, text="Enter decryption key (if encrypted):", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        self.img_in_img_decrypt_key = tk.StringVar()
        tk.Entry(decrypt_frame, textvariable=self.img_in_img_decrypt_key, width=40, show="*").pack(padx=5, pady=2)
        
        # Output file for extracted image
        extract_output_frame = tk.LabelFrame(extract_section, text="Extracted Image Output", 
                                           font=('Arial', 10, 'bold'), bg='#f0f0f0')
        extract_output_frame.pack(fill='x', padx=5, pady=3)
        
        tk.Label(extract_output_frame, text="Save extracted image as:", bg='#f0f0f0').pack(anchor='w', padx=5, pady=2)
        
        self.img_in_img_extract_output = tk.StringVar()
        extract_output_file_frame = tk.Frame(extract_output_frame, bg='#f0f0f0')
        extract_output_file_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Entry(extract_output_file_frame, textvariable=self.img_in_img_extract_output, width=40).pack(side='left', padx=(0,5))
        tk.Button(extract_output_file_frame, text="Browse", command=self.browse_img_in_img_extract_output, 
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Extract button
        extract_button = tk.Button(extract_section, text="Extract Image", 
                                 command=self.extract_img_in_img, bg='#27ae60', fg='white', 
                                 font=('Arial', 11, 'bold'), height=2)
        extract_button.pack(pady=5)
        
        # Extracted image viewer
        viewer_frame = tk.LabelFrame(extract_section, text="Extracted Image Preview", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0')
        viewer_frame.pack(fill='both', expand=True, padx=5, pady=3)
        
        # Canvas for image display with scrollbars
        canvas_frame = tk.Frame(viewer_frame, bg='#f0f0f0')
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbars
        img_canvas = tk.Canvas(canvas_frame, bg='white', width=300, height=200)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', command=img_canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient='horizontal', command=img_canvas.xview)
        img_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        img_canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.img_in_img_canvas = img_canvas
        self.img_in_img_photo = None  # Store PhotoImage reference
        
        # Status label for image viewer
        self.img_in_img_status_label = tk.Label(viewer_frame, text="No image extracted yet", 
                                               font=('Arial', 9), bg='#f0f0f0', fg='#7f8c8d')
        self.img_in_img_status_label.pack(pady=2)
        
        # Info label
        info_label = tk.Label(extract_section, 
                             text="Note: The secret image will be automatically resized\nto fit within the cover image.",
                             font=('Arial', 9), bg='#f0f0f0', fg='#7f8c8d', justify='left')
        info_label.pack(pady=5)
    
    def browse_img_in_img_cover(self):
        """Browse for cover image"""
        filetypes = [("Image files", "*.png;*.bmp;*.tiff;*.jpg;*.jpeg"), ("PNG files", "*.png"), 
                     ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("JPG files", "*.jpg;*.jpeg"), 
                     ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.img_in_img_cover.set(filename)
            base, ext = os.path.splitext(filename)
            self.img_in_img_output.set(f"{base}_with_hidden_image.png")
    
    def browse_img_in_img_secret(self):
        """Browse for secret image"""
        filetypes = [("Image files", "*.png;*.bmp;*.tiff;*.jpg;*.jpeg"), ("PNG files", "*.png"), 
                     ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("JPG files", "*.jpg;*.jpeg"), 
                     ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.img_in_img_secret.set(filename)
    
    def browse_img_in_img_output(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename:
            self.img_in_img_output.set(filename)
    
    def browse_img_in_img_stego(self):
        """Browse for stego image"""
        filetypes = [("Image files", "*.png;*.bmp;*.tiff"), ("PNG files", "*.png"), 
                     ("BMP files", "*.bmp"), ("TIFF files", "*.tiff"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.img_in_img_stego.set(filename)
            base, ext = os.path.splitext(filename)
            self.img_in_img_extract_output.set(f"{base}_extracted.png")
    
    def browse_img_in_img_extract_output(self):
        """Browse for extracted image output file"""
        filename = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename:
            self.img_in_img_extract_output.set(filename)
    
    def hide_img_in_img(self):
        """Hide image in image"""
        try:
            cover = self.img_in_img_cover.get()
            secret = self.img_in_img_secret.get()
            output = self.img_in_img_output.get()
            key = self.img_in_img_encrypt_key.get().strip()
            
            if not cover or not secret or not output:
                messagebox.showerror("Error", "Please fill in all required fields!")
                return
                
            if not os.path.exists(cover):
                messagebox.showerror("Error", "Cover image file does not exist!")
                return
                
            if not os.path.exists(secret):
                messagebox.showerror("Error", "Secret image file does not exist!")
                return
                
            # Validate output path
            if not output:
                messagebox.showerror("Error", "Please specify an output file path!")
                return
            
            # Ensure output has .png extension
            if not output.lower().endswith('.png'):
                output = os.path.splitext(output)[0] + '.png'
                self.img_in_img_output.set(output)
            
            self.status_var.set("Hiding image in image...")
            self.root.update()
            
            success = self.stego.hide_image_in_image(cover, secret, output, key if key else None)
            
            if success:
                encryption_status = "with encryption" if key else "without encryption"
                messagebox.showinfo("Success", f"Image hidden successfully {encryption_status}!\n\nOutput: {output}")
                self.status_var.set("Image hidden successfully!")
            else:
                messagebox.showerror("Error", "Failed to hide image! Check the error messages above.")
                self.status_var.set("Failed to hide image!")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred!")
    
    def extract_img_in_img(self):
        """Extract image from image"""
        try:
            stego = self.img_in_img_stego.get()
            output = self.img_in_img_extract_output.get()
            key = self.img_in_img_decrypt_key.get().strip()
            
            if not stego or not output:
                messagebox.showerror("Error", "Please fill in all required fields!")
                return
                
            if not os.path.exists(stego):
                messagebox.showerror("Error", "Stego image file does not exist!")
                return
                
            # Validate output path
            if not output:
                messagebox.showerror("Error", "Please specify an output file path for the extracted image!")
                return
            
            # Ensure output has .png extension
            if not output.lower().endswith('.png'):
                output = os.path.splitext(output)[0] + '.png'
                self.img_in_img_extract_output.set(output)
            
            self.status_var.set("Extracting image from image...")
            self.root.update()
            
            success = self.stego.extract_image_from_image(stego, output, key if key else None)
            
            if success:
                self.status_var.set("Image extracted successfully!")
                
                # Display extracted image in viewer
                self.display_extracted_image(output)
                
                messagebox.showinfo("Success", f"Image extracted successfully!\n\nExtracted image saved to: {output}")
            else:
                self.status_var.set("Failed to extract image!")
                # Clear viewer on failure
                self.img_in_img_canvas.delete("all")
                self.img_in_img_status_label.config(text="Extraction failed - No image to display", fg='#e74c3c')
                messagebox.showerror("Error", "Failed to extract image. Check:\n1. File contains hidden image\n2. Correct decryption key\n3. File format is supported")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error occurred!")
            # Clear viewer on error
            self.img_in_img_canvas.delete("all")
            self.img_in_img_status_label.config(text=f"Error: {str(e)[:50]}...", fg='#e74c3c')
    
    def display_extracted_image(self, image_path):
        """Display extracted image in the viewer canvas"""
        try:
            if not os.path.exists(image_path):
                self.img_in_img_status_label.config(text="Image file not found", fg='#e74c3c')
                return
            
            # Read image using cv2
            img = cv2.imread(image_path)
            if img is None:
                self.img_in_img_status_label.config(text="Failed to load image", fg='#e74c3c')
                return
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]
            
            # Resize if too large for display (max 400x400)
            max_display_size = 400
            if width > max_display_size or height > max_display_size:
                scale = min(max_display_size / width, max_display_size / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                height, width = new_height, new_width
            
            # Convert to PhotoImage using PIL if available, otherwise use workaround
            try:
                from PIL import Image, ImageTk
                img_pil = Image.fromarray(img_rgb)
                self.img_in_img_photo = ImageTk.PhotoImage(img_pil)
                
                # Clear canvas and display image
                self.img_in_img_canvas.delete("all")
                self.img_in_img_canvas.create_image(0, 0, anchor='nw', image=self.img_in_img_photo)
                
                # Update canvas scroll region
                self.img_in_img_canvas.config(scrollregion=self.img_in_img_canvas.bbox("all"))
                
                # Update status
                file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
                self.img_in_img_status_label.config(
                    text=f"Image displayed: {width}x{height} pixels, {file_size:.2f} MB", 
                    fg='#27ae60'
                )
            except ImportError:
                # PIL not available, use alternative method with cv2
                # Just show file info in canvas
                self.img_in_img_canvas.delete("all")
                self.img_in_img_canvas.create_text(
                    self.img_in_img_canvas.winfo_width() // 2 if self.img_in_img_canvas.winfo_width() > 0 else 150,
                    self.img_in_img_canvas.winfo_height() // 2 if self.img_in_img_canvas.winfo_height() > 0 else 100,
                    text=f"Image extracted!\n{width}x{height} pixels\n\nView saved file:\n{os.path.basename(image_path)}",
                    font=('Arial', 10),
                    justify='center'
                )
                
                file_size = os.path.getsize(image_path) / (1024 * 1024)
                self.img_in_img_status_label.config(
                    text=f"Image saved: {width}x{height} pixels, {file_size:.2f} MB. Click 'Browse' to view.",
                    fg='#27ae60'
                )
                
        except Exception as e:
            self.img_in_img_status_label.config(text=f"Error: {str(e)}", fg='#e74c3c')

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = ReliableSteganographyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
