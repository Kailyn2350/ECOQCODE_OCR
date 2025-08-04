---
marp: true
theme: uncover
size: 16:9
paginate: true
transition: fade
---

<!-- _class: lead -->
<!-- _paginate: false -->

# **ECOQCODE OCR**
### 🖼️ 이미지 내 특정 텍스트 검출 및 인식 시스템

**발표자:** (이름을 입력하세요)
**날짜:** 2025년 7월 30일

---

<!-- _class: invert -->

## **프로젝트 개요 (Introduction)**

*   **🤔 문제 정의**
    *   수많은 이미지 속에서 'ECOQCODE'라는 특정 텍스트의 존재 여부를 **자동으로 판별**해야 할 필요성 대두
*   **🎯 프로젝트 목표**
    *   이미지 내 'ECOQCODE' 텍스트 유무를 판별하는 **분류(Classification)** 모델 개발
    *   텍스트를 정확히 읽어내는 **OCR(Optical Character Recognition)** 모델 개발
*   **🛠️ 핵심 기술**
    *   합성 데이터 생성 (Synthetic Data Generation)
    *   딥러닝 기반 모델 (CNN, CRNN)
    *   ONNX를 통한 모델 배포 및 경량화

---

## **프로젝트 특징 및 주요 기능 (Features)**

*   **✨ 자동화된 데이터셋 구축**
    *   다양한 배경과 폰트 크기를 활용하여 'ECOQCODE' 텍스트가 포함된 합성 데이터를 대량으로 생성
*   **🧠 딥러닝 모델 구현**
    *   **1단계: 텍스트 유무 판별 (Classifier)**
        *   이미지 분류를 위한 맞춤형 CNN(Convolutional Neural Network) 아키텍처 사용
    *   **2단계: 텍스트 인식 (OCR)**
        *   CRNN(Convolutional Recurrent Neural Network) 모델을 활용하여 이미지 속 문자열을 인식
*   **🚀 모델 배포 및 활용**
    *   학습된 모델을 ONNX(Open Neural Network Exchange) 형식으로 변환
    *   플랫폼에 제약 없이 모델을 쉽게 배포하고 추론에 활용 가능

---

## **개발 과정 (Workflow)**

<style scoped>
.workflow-container {
    display: flex;
    justify-content: space-around;
    align-items: center;
    height: 100%;
}
.workflow-item {
    text-align: center;
    width: 20%;
}
.arrow {
    font-size: 3rem;
    color: #888;
}
</style>

<div class="workflow-container">
    <div class="workflow-item">
        <h3>1. 🖼️</h3>
        <h4>데이터 생성</h4>
        <p>배경 이미지에<br>텍스트 합성</p>
    </div>
    <div class="arrow">➡️</div>
    <div class="workflow-item">
        <h3>2. 🧠</h3>
        <h4>모델 학습</h4>
        <p>CNN & CRNN<br>모델 학습</p>
    </div>
    <div class="arrow">➡️</div>
    <div class="workflow-item">
        <h3>3. 📊</h3>
        <h4>모델 평가</h4>
        <p>성능 검증 및<br>지표 확인</p>
    </div>
    <div class="arrow">➡️</div>
    <div class="workflow-item">
        <h3>4. 🚀</h3>
        <h4>모델 변환</h4>
        <p>PyTorch 모델을<br>ONNX로 Export</p>
    </div>
</div>

---

## **기술 상세 1 - 데이터 생성**

*   **❓ 왜 합성 데이터를 사용했는가?**
    *   실제 'ECOQCODE' 이미지 데이터를 대량으로 수집하기 어려운 문제를 해결
    *   다양한 환경(배경, 조명, 폰트)에 강건한(Robust) 모델을 만들기 위함
*   **⚙️ 생성 과정**
    *   `Pillow` 라이브러리를 사용하여 `real_backgrounds/`의 이미지 위에 `arial.ttf` 폰트로 텍스트 렌더링
    *   'ECOQCODE'가 포함된 이미지(Positive)와 포함되지 않은 이미지(Negative)를 생성하여 데이터셋 균형 유지

---

<!-- _header: '**기술 상세 2 - 딥러닝 모델**' -->

<div style="display: flex; gap: 2rem; margin-top: 2rem;">
<div style="flex: 1; border: 1px solid #ddd; padding: 1rem; border-radius: 10px;">

### **1. ECOQCODE 분류 모델 (CNN)**
*   **역할**: 이미지에 'ECOQCODE'가 있는지 없는지(Yes/No)를 판단
*   **구조**: 여러 개의 합성곱(Convolution) 레이어와 풀링(Pooling) 레이어로 구성된 간단하고 효율적인 이미지 분류 모델
*   **산출물**: `ecoq_classifier.onnx`

</div>
<div style="flex: 1; border: 1px solid #ddd; padding: 1rem; border-radius: 10px;">

### **2. ECOQCODE 인식 모델 (CRNN)**
*   **역할**: 이미지 속에서 'ECOQCODE'라는 글자를 정확히 읽어냄
*   **구조**: CNN으로 이미지 특징을 추출하고, RNN(LSTM)으로 문자열의 순차적 특징을 학습
*   **산출물**: `ecoq_crnn_ocr.onnx`

</div>
</div>

---

## **기술 상세 3 - ONNX 변환 및 배포**

<div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin-top: 3rem;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pytorch_logo.svg/1200px-Pytorch_logo.svg.png" width="200">
    <span style="font-size: 4rem;">➡️</span>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/ONNX_logo_main.svg/1200px-ONNX_logo_main.svg.png" width="200">
</div>

*   **ONNX (Open Neural Network Exchange)란?**
    *   서로 다른 딥러닝 프레임워크 간에 모델을 공유하고 배포할 수 있는 개방형 표준
*   **ONNX 변환의 장점**
    *   **🤝 상호 운용성**: PyTorch로 학습하고 C#, Java, Python 등 다양한 환경에서 사용 가능
    *   **⚡ 성능 최적화**: ONNX Runtime을 통해 하드웨어 가속을 활용하여 더 빠른 추론 속도 확보
    *   **📦 경량화**: 모델을 배포하기 용이한 단일 파일(`.onnx`)로 패키징

---

<!-- _class: invert -->

## **프로젝트 성과 및 향후 개선 방향**

*   **📈 주요 성과**
    *   'ECOQCODE' 텍스트 유무를 판별하는 자동화된 시스템 구축 완료
    *   합성 데이터 생성을 통해 데이터 부족 문제 해결 및 모델 강건성 확보
    *   ONNX 모델을 통해 실제 서비스에 적용 가능한 수준의 배포 용이성 확보
*   **🔭 향후 개선 방향**
    *   더 다양한 실제 배경 이미지와 폰트를 활용하여 데이터셋 고도화
    *   전이 학습(Transfer Learning) 등 더 복잡한 모델 아키텍처를 도입하여 성능 향상 모색
    *   ONNX 모델을 실제 애플리케이션(웹, 모바일 등)에 통합하는 후속 프로젝트 진행

---

<!-- _class: lead -->
<!-- _paginate: false -->

## **Q&A**

### **질의응답**