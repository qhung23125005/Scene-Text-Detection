# Scene Text Recognition

This project is part of **Module 6** of the course **AIO24**. It includes **text detection**, **text recognition**, and **deployment-ready code** to run the system as a service.

---

## Features

- **YOLOv11** for accurate and fast **text region detection**
- **CRNN** for robust **character-level text recognition**
- ICDAR2003 dataset support with automated preprocessing
- Full training + evaluation pipeline for both detection and recognition
- Ready for deployment using **Ray Serve** and **Streamlit**

## ⚙️ Model Details

### 🔍 Text Detection
- Use YOLOv11
- Pretrained and fine-tuned on converted ICDAR2003 dataset
- The weights of the model is saved in /runs/content/runs/detect/train/weights/best.pt

### 🔤 Text Recognition (CRNN)
- Implement in Pytorch
- Use CRNN and train on ICDAR2003 dataset
- The weights of the model is saved in ocr_crnn.pt

---
## Deployment
1. Create and Activate virtual environment (optional). Download the requirements

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd deployment
```

2. Activate Ray
```bash
make init
```

3. Run the server
```bash
make deploy_ocr
```

4. Run streamlit
```bash
make streamlit
```

## Results

The results are shown below:

![image](https://github.com/user-attachments/assets/d14df393-143e-4c30-8b1a-9c322d2b2555)
*Figure 1: Deploy using Ray Serve.*



![image](https://github.com/user-attachments/assets/42fffcf9-02d1-46f2-a329-c832373a486b)
*Figure 2: Streamlit*




