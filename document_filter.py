import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def median_blur_denoise(image, kernel_size=5):
    """
    Apply median blur to remove noise from document
    
    Args:
        image: Input image
        kernel_size: Size of median filter kernel (must be odd, 3, 5, 7, etc.)
    
    Returns:
        Denoised image
    """
    # Apply median blur
    denoised = cv2.medianBlur(image, kernel_size)
    return denoised


def advanced_document_cleanup(image):
    """
    Advanced noise removal with multiple techniques
    
    Args:
        image: Input document image
    
    Returns:
        Cleaned image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Apply median blur for salt-and-pepper noise
    denoised = cv2.medianBlur(gray, 5)
    
    # 2. Apply morphological operations to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    # 3. Apply adaptive thresholding for better text contrast
    threshold = cv2.adaptiveThreshold(
        cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return threshold


def remove_document_noise(image_path, output_path='filtered_document.jpg'):
    """
    Remove noise from old document image using median blur
    
    Args:
        image_path: Path to the input document image
        output_path: Path to save the filtered image
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return None
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    print(f"✓ Successfully loaded document: {image_path}")
    print(f"  Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Convert to grayscale for document processing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        gray = img.copy()
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    print("\nApplying noise removal techniques...")
    
    # Apply different levels of median blur
    mild_blur = median_blur_denoise(gray, kernel_size=3)
    medium_blur = median_blur_denoise(gray, kernel_size=5)
    strong_blur = median_blur_denoise(gray, kernel_size=7)
    
    # Apply advanced cleanup
    advanced_clean = advanced_document_cleanup(img)
    
    # Apply bilateral filter (preserves edges better)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply non-local means denoising
    nlm_denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Display results
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Document Noise Removal - Median Blur Technique', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Original
    plt.subplot(2, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original (Noisy)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Mild median blur
    plt.subplot(2, 4, 2)
    plt.imshow(mild_blur, cmap='gray')
    plt.title('Median Blur (3x3)\nMild', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Medium median blur
    plt.subplot(2, 4, 3)
    plt.imshow(medium_blur, cmap='gray')
    plt.title('Median Blur (5x5) ⭐\nRecommended', fontsize=12, fontweight='bold', color='green')
    plt.axis('off')
    
    # Strong median blur
    plt.subplot(2, 4, 4)
    plt.imshow(strong_blur, cmap='gray')
    plt.title('Median Blur (7x7)\nStrong', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Advanced cleanup
    plt.subplot(2, 4, 5)
    plt.imshow(advanced_clean, cmap='gray')
    plt.title('Advanced Cleanup\n(Median + Morphology + Threshold)', 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Bilateral filter
    plt.subplot(2, 4, 6)
    plt.imshow(bilateral, cmap='gray')
    plt.title('Bilateral Filter\n(Edge Preserving)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Non-local means
    plt.subplot(2, 4, 7)
    plt.imshow(nlm_denoised, cmap='gray')
    plt.title('Non-Local Means\nDenoising', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Comparison zoom (center crop)
    h, w = gray.shape[:2]
    crop_size = min(h, w) // 4
    y1, y2 = h//2 - crop_size//2, h//2 + crop_size//2
    x1, x2 = w//2 - crop_size//2, w//2 + crop_size//2
    
    # Create side-by-side comparison
    comparison = np.hstack([
        gray[y1:y2, x1:x2],
        medium_blur[y1:y2, x1:x2]
    ])
    
    plt.subplot(2, 4, 8)
    plt.imshow(comparison, cmap='gray')
    plt.title('Detail Comparison\nOriginal (Left) | Filtered (Right)', 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('noise_removal_comparison.jpg', dpi=150, bbox_inches='tight')
    print("✓ Comparison saved as: noise_removal_comparison.jpg")
    plt.show()
    
    # Save the filtered results
    # Main output - medium median blur (best balance)
    cv2.imwrite(output_path, medium_blur)
    print(f"✓ Filtered document saved as: {output_path}")
    
    # Save advanced version
    advanced_output = output_path.replace('.jpg', '_advanced.jpg')
    cv2.imwrite(advanced_output, advanced_clean)
    print(f"✓ Advanced filtered document saved as: {advanced_output}")
    
    # Save bilateral version
    bilateral_output = output_path.replace('.jpg', '_bilateral.jpg')
    cv2.imwrite(bilateral_output, bilateral)
    print(f"✓ Bilateral filtered document saved as: {bilateral_output}")
    
    # Calculate and display noise reduction metrics
    print("\n" + "="*60)
    print("NOISE REMOVAL METRICS")
    print("="*60)
    print(f"Original Image Shape: {gray.shape}")
    print(f"Filtered Image Shape: {medium_blur.shape}")
    print(f"Median Blur Parameters: Kernel Size=5x5")
    print(f"Filter Type: Salt-and-Pepper Noise Removal")
    
    # Calculate image statistics
    original_mean = np.mean(gray)
    original_std = np.std(gray)
    filtered_mean = np.mean(medium_blur)
    filtered_std = np.std(medium_blur)
    
    print(f"\nImage Statistics:")
    print(f"  Original  - Mean: {original_mean:.2f}, Std Dev: {original_std:.2f}")
    print(f"  Filtered  - Mean: {filtered_mean:.2f}, Std Dev: {filtered_std:.2f}")
    print(f"  Noise Reduction: {((original_std - filtered_std) / original_std * 100):.2f}%")
    print("="*60)
    
    return medium_blur, advanced_clean, bilateral


def batch_process_documents(input_folder='image_file', output_folder='filtered_output'):
    """
    Process multiple document images to remove noise
    
    Args:
        input_folder: Folder containing document images
        output_folder: Folder to save filtered images
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✓ Created output folder: {output_folder}")
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Get all image files
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return
    
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: Found {len(image_files)} documents")
    print("="*60)
    
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"filtered_{filename}"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
        
        # Read and filter
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            filtered = median_blur_denoise(img, kernel_size=5)
            cv2.imwrite(output_path, filtered)
            print(f"  ✓ Saved: {output_path}")
        else:
            print(f"  ✗ Failed to load: {filename}")
    
    print(f"\n{'='*60}")
    print(f"✓ Batch processing complete!")
    print(f"  Output folder: {output_folder}")
    print("="*60)


def main():
    print("="*60)
    print("   DOCUMENT NOISE REMOVAL - MEDIAN BLUR TECHNIQUE")
    print("="*60)
    print("\nDomain: Document Processing")
    print("Object: Document")
    print("Technique: Median Blur")
    print("Output: Filtered Image")
    print("="*60)
    
    # Set your document image path here
    image_path = r'image_file\document.jpg'
    
    # Try alternative paths if not found
    if not os.path.exists(image_path):
        possible_paths = [
            r'document.jpg',
            r'image_file\document.jpg',
            r'images\document.jpg',
            r'IMAGE_FILE\document.jpg',
            r'image_file\old_document.jpg',
        ]
        
        print(f"\nSearching for document image...")
        found = False
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✓ Found document at: {path}")
                image_path = path
                found = True
                break
        
        if not found:
            print("\n❌ Could not find document image!")
            print(f"Current directory: {os.getcwd()}")
            print("\nPlease place your document image in one of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nExpected directory structure:")
            print("C:\\removing_noise\\")
            print("├── document_filter.py")
            print("└── image_file\\")
            print("    └── document.jpg")
            return
    
    # Remove noise from document
    print(f"\nProcessing document: {image_path}")
    result = remove_document_noise(image_path, output_path='filtered_document.jpg')
    
    if result is not None:
        print("\n" + "="*60)
        print("✓ SUCCESS! Document noise removal complete!")
        print("="*60)
        print("\nOutput files generated:")
        print("  1. filtered_document.jpg - Median blur (recommended)")
        print("  2. filtered_document_advanced.jpg - Advanced cleanup")
        print("  3. filtered_document_bilateral.jpg - Edge preserving")
        print("  4. noise_removal_comparison.jpg - Visual comparison")
        print("="*60)
        
        # Ask if user wants batch processing
        print("\n💡 TIP: To process multiple documents, use:")
        print("   batch_process_documents('image_file', 'filtered_output')")


if __name__ == "__main__":
    main()