"""
Quick test script to verify crop functionality
"""
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# Import the functions to test
from convert_data import _detect_defect_regions, _calculate_crop_box, _crop_image_and_mask


def create_test_mask_with_multiple_defects():
    """Create a test mask image with 3 separate defect regions (similar to the example)"""
    # Create a 1920x1080 black image
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    
    # Add 3 white defect regions at different locations
    # Region 1: top-left
    mask[100:300, 150:350] = 255
    
    # Region 2: top-right  
    mask[200:400, 1500:1700] = 255
    
    # Region 3: bottom-center
    mask[700:900, 800:1000] = 255
    
    # Convert to bytes
    img = Image.fromarray(mask)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def create_test_image():
    """Create a test RGB image"""
    # Create a colored test image
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def test_defect_detection():
    """Test defect region detection"""
    print("=" * 60)
    print("Testing defect detection...")
    print("=" * 60)
    
    mask_bytes = create_test_mask_with_multiple_defects()
    regions = _detect_defect_regions(mask_bytes, min_area=100)
    
    print(f"✓ Detected {len(regions)} defect region(s)")
    for i, region in enumerate(regions):
        x_min, y_min, x_max, y_max = region
        w, h = x_max - x_min, y_max - y_min
        print(f"  Region {i+1}: bbox=({x_min}, {y_min}, {x_max}, {y_max}), size=({w}x{h})")
    
    assert len(regions) == 3, f"Expected 3 regions, got {len(regions)}"
    print("✓ Detection test passed!\n")
    return regions


def test_crop_box_calculation(regions):
    """Test crop box calculation"""
    print("=" * 60)
    print("Testing crop box calculation...")
    print("=" * 60)
    
    img_size = (1920, 1080)
    
    for i, defect_bbox in enumerate(regions):
        crop_box = _calculate_crop_box(
            defect_bbox,
            img_size,
            min_crop_size=768,
            padding_ratio=0.1
        )
        x_min, y_min, x_max, y_max = crop_box
        w, h = x_max - x_min, y_max - y_min
        
        print(f"  Region {i+1}:")
        print(f"    Defect bbox: {defect_bbox}")
        print(f"    Crop box: {crop_box}")
        print(f"    Crop size: {w}x{h}")
        
        # Verify crop size is at least 768 (or limited by image size)
        assert w >= min(768, img_size[0]), f"Width {w} < 768"
        assert h >= min(768, img_size[1]), f"Height {h} < 768"
        
        # Verify crop box is within image bounds
        assert x_min >= 0 and x_max <= img_size[0], "Crop box exceeds image width"
        assert y_min >= 0 and y_max <= img_size[1], "Crop box exceeds image height"
    
    print("✓ Crop box calculation test passed!\n")
    return regions


def test_crop_execution(regions):
    """Test actual cropping of image and mask"""
    print("=" * 60)
    print("Testing image and mask cropping...")
    print("=" * 60)
    
    img_bytes = create_test_image()
    mask_bytes = create_test_mask_with_multiple_defects()
    img_size = (1920, 1080)
    
    for i, defect_bbox in enumerate(regions):
        crop_box = _calculate_crop_box(
            defect_bbox,
            img_size,
            min_crop_size=768,
            padding_ratio=0.1
        )
        
        cropped_img_bytes, cropped_mask_bytes = _crop_image_and_mask(
            img_bytes,
            mask_bytes,
            crop_box,
            compress=True,
            quality=85
        )
        
        # Verify output is valid
        cropped_img = Image.open(BytesIO(cropped_img_bytes))
        cropped_mask = Image.open(BytesIO(cropped_mask_bytes))
        
        print(f"  Region {i+1}:")
        print(f"    Cropped image size: {cropped_img.size}")
        print(f"    Cropped mask size: {cropped_mask.size}")
        print(f"    Cropped image mode: {cropped_img.mode}")
        print(f"    Cropped mask mode: {cropped_mask.mode}")
        
        # Verify sizes match
        assert cropped_img.size == cropped_mask.size, "Image and mask sizes don't match"
        
        # Verify image is RGB
        assert cropped_img.mode == "RGB", f"Image mode should be RGB, got {cropped_img.mode}"
        
        # Verify mask is grayscale
        assert cropped_mask.mode in ("L", "1"), f"Mask mode should be L or 1, got {cropped_mask.mode}"
    
    print("✓ Cropping test passed!\n")


def main():
    print("\n" + "=" * 60)
    print("CROP FUNCTIONALITY TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Defect detection
        regions = test_defect_detection()
        
        # Test 2: Crop box calculation
        test_crop_box_calculation(regions)
        
        # Test 3: Actual cropping
        test_crop_execution(regions)
        
        print("=" * 60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nThe crop functionality is working correctly!")
        print("You can now run convert_data.py on your actual dataset.\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("=" * 60)
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

