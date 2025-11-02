# removing_noise
# 📄 Document Noise Removal and Cleanup

This project implements various image processing techniques using OpenCV to clean up noisy document images, often encountered with old paper documents or poor scans.

The primary technique demonstrated is **Median Blur** for removing salt-and-pepper noise, but the script also compares it with **Advanced Thresholding**, **Bilateral Filtering**, and **Non-Local Means Denoising**.

| Detail | Value |
| :--- | :--- |
| **Domain** | Document Processing |
| **Object** | Document Image |
| **Technique** | Median Blur, Advanced Cleanup |
| **Output** | Filtered Image (Grayscale/Binary) |

---

## ⚙️ Prerequisites

You must have Python installed. Install the necessary libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
