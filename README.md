# autoControlBasedOnThreeAxisMechine

## Image Enhancement for OCR

To improve OCR accuracy, the application includes an advanced image enhancement pipeline applied to screenshots before they are sent to the OCR engine. This pipeline is configurable via `config.json` (or `DEFAULT_CONFIG` in `main.py`).

The enhancement steps are applied in the following order:

1.  **Deskewing (Optional)**:
    *   Corrects the orientation of skewed or tilted text in the image.
    *   Configuration:
        *   `ENHANCE_IMAGE_DESKEWING` (boolean): Set to `true` to enable, `false` to disable. Default: `false`.

2.  **Noise Reduction (Optional)**:
    *   Applies a Gaussian blur to reduce image noise.
    *   Configuration:
        *   `ENHANCE_IMAGE_NOISE_REDUCTION` (boolean): Set to `true` to enable. Default: `true`.
        *   `ENHANCE_IMAGE_GAUSSIAN_BLUR_KERNEL_SIZE` (array of two odd integers, e.g., `[5, 5]`): Kernel size for the Gaussian blur. Default: `[5, 5]`.

3.  **Contrast Limited Adaptive Histogram Equalization (CLAHE)**:
    *   Enhances the contrast of the image. This step is always applied.

4.  **Sharpening (Optional)**:
    *   Applies a sharpening filter to enhance text edges. Use with caution, as it can amplify noise.
    *   Configuration:
        *   `ENHANCE_IMAGE_SHARPENING` (boolean): Set to `true` to enable. Default: `false`.
        *   `ENHANCE_IMAGE_SHARPENING_KERNEL` (string representation of a 2D numpy array): The kernel used for sharpening. Default: `"[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]"`.

5.  **Adaptive Thresholding (Optional)**:
    *   Converts the image to black and white, which can significantly improve OCR.
    *   Configuration:
        *   `ENHANCE_IMAGE_ADAPTIVE_THRESHOLDING` (boolean): Set to `true` to enable. Default: `true`.
        *   `ENHANCE_IMAGE_ADAPTIVE_THRESHOLDING_METHOD` (string): Method to use - `"gaussian"` or `"mean"`. Default: `"gaussian"`.
        *   `ENHANCE_IMAGE_ADAPTIVE_THRESHOLDING_BLOCK_SIZE` (odd integer): Size of the pixel neighborhood used to calculate the threshold. Default: `11`.
        *   `ENHANCE_IMAGE_ADAPTIVE_THRESHOLDING_C` (integer): Constant subtracted from the mean or weighted mean. Default: `2`.

By tuning these parameters, you can optimize the image preprocessing for your specific OCR needs and image characteristics.