# OCR models comparison

## Project Overview

This project evaluates and compares the performance of 5 different OCR (Optical Character Recognition) models on English and French datasets. The goal is to provide comprehensive analysis of accuracy, speed, multilingual support, ease of integration, cost, and license/infrastructure requirements for each model.

## OCR models evalulated

1. **Google Vision API** - Cloud OCR service by Google
2. **Tesseract** - Open-source OCR engine, traditional approach
3. **EasyOCR** - Deep learning-based OCR
4. **DocTR** - Also a deeplearning based OCR
5. **Surya** - Modern OCR with detection and recognition, also deep learning

## Dataset

- **English dataset**: 10 samples (bills/transcripts)
- **French dataset**: 10 samples (book pages)
- **Total**: 20 samples
- **Formats**: Images with annotations (JSON for English, XML for French)

## Evaluation Criteria

### 1. Accuracy

#### Text Recognition Metrics:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Accuracy of predicted words
- **Recall**: Coverage of ground truth words
- **Similarity**: Text similarity score

#### Spatial Detection Metrics (for the 4 models that support it, tesseract did not support bbox):
- **F1**: Bounding box detection accuracy
- **avg IoU**: Intersection over Union for detected regions

### 2. Speed
- **Processing Time**: Average time per image (seconds)

### 3. Multilingual Support
- Tested on English and French datasets

### 4. Ease of Integration
- Installation complexity
- API simplicity
- Dependencies requirements

### 5. Cost
- Free vs. paid services
- Usage limitations
- Infrastructure costs

### 6. License/Infrastructure Requirements
- Open source vs. proprietary
- Cloud vs. local deployment
- Hardware requirements

## Evaluation Results

### Performance Summary
- Results with only using text
![text](./screenshots/text.png)

- Results using both text and bounding box
![spatial](./screenshots/spatial.png)

## Detailed Analysis

### 1. Accuracy Analysis

**Google Vision API** - Highest overall accuracy
- Achieves the best F1 score (0.803) and precision (0.807), demonstrating superior text recognition capabilities
- The high precision indicates fewer false positives, making it reliable for production environments
- Good similarity score (0.456) shows strong semantic understanding of text content
- Leverages Google's massive training datasets and proprietary deep learning models
- Excels at handling various document types, fonts, and image qualities
- However, spatial detection performance is weaker (Detection F1: 0.064) compared to specialized document analysis models

**DocTR** - Strong second place with excellent spatial understanding
- Achieves solid F1 score (0.778) with well-balanced precision (0.774) and recall (0.786)
- Best spatial detection capabilities among all models (IoU: 0.671), making it ideal for document layout analysis
- The balanced precision/recall ratio indicates consistent performance across different text types
- Designed specifically for document processing, showing superior understanding of document structure
- Lower text similarity score (0.247) suggests it may struggle with semantic text understanding compared to Google Vision
- Represents the best trade-off between accuracy and spatial awareness for open-source solutions

**Surya** - Competitive modern approach
- Decent F1 score (0.767) with good precision (0.761) and recall (0.776) balance
- Best detection F1 (0.572) among open-source models, showing strong bounding box detection capabilities
- Modern transformer-based architecture provides good multilingual support
- Reasonable spatial detection performance (IoU: 0.649) makes it suitable for document analysis tasks
- Lower text similarity (0.201) indicates room for improvement in semantic understanding
- Represents cutting-edge open-source OCR technology with active development

**Tesseract** - Traditional but reliable
- Moderate F1 score (0.638) with balanced precision (0.633) and recall (0.649)
- Higher similarity score (0.320) than some deep learning models, showing good text understanding
- Mature, well-tested technology with decades of development and optimization
- Performance limitations on complex layouts and modern document formats
- No spatial detection capabilities, limiting its use for document structure analysis
- Best suited for simple, clean text extraction tasks

**EasyOCR** - Accessible but limited
- Lowest F1 score (0.593) among all models, indicating accuracy challenges
- Balanced precision (0.593) and recall (0.595) but overall lower performance
- Moderate spatial detection capabilities (Detection F1: 0.286, IoU: 0.690)
- Designed for ease of use rather than maximum accuracy
- Good for quick prototyping and simple OCR tasks
- May struggle with complex document layouts and challenging text conditions

### 2. Speed Analysis

**Google Vision API** - Fastest processing (0.67s)
- Cloud-based processing leverages Google's optimized infrastructure and specialized hardware (TPUs)
- Speed advantage comes from distributed computing and pre-optimized models
- Performance depends on internet connectivity and network latency
- Consistent processing times regardless of local hardware capabilities
- Ideal for applications requiring fast processing with reliable internet connection
- May not be suitable for real-time applications with strict latency requirements due to network dependency

**Tesseract** - Fast local processing (4.33s)
- CPU-based processing makes it hardware-agnostic and suitable for resource-constrained environments
- Traditional computer vision algorithms are computationally efficient
- Consistent performance across different hardware configurations
- No network dependency ensures reliable processing times
- Suitable for real-time applications and offline processing
- Performance scales with CPU capabilities but remains relatively fast

**Surya** - Moderate processing time (7.02s)
- Modern deep learning architecture requires more computational resources than traditional methods
- GPU acceleration can significantly improve processing times
- Transformer-based models have higher computational overhead but provide better accuracy
- Processing time varies significantly based on available hardware (CPU vs GPU)
- Represents good balance between accuracy and speed for modern OCR applications

**EasyOCR** - Reasonable processing time (5.16s)
- Deep learning-based but optimized for accessibility and ease of use
- Moderate computational requirements make it suitable for various hardware configurations
- GPU acceleration available but not strictly required
- Processing time reflects the balance between accuracy and computational efficiency
- Good choice for applications where moderate speed and ease of use are priorities

**DocTR** - Slowest processing (11.31s)
- Highest computational requirements due to sophisticated deep learning architecture
- Complex document analysis pipeline includes both detection and recognition stages
- GPU acceleration strongly recommended for acceptable performance
- Processing time reflects the comprehensive document understanding capabilities
- Trade-off between speed and advanced document analysis features
- Best suited for batch processing rather than real-time applications

### 3. Multilingual Support Analysis

**Google Vision API** - Comprehensive language support
- Supports 100+ languages with excellent performance across major language families
- Leverages Google's massive multilingual training data and advanced language models
- Consistent performance across different scripts (Latin, Cyrillic, Asian characters)
- Automatic language detection capabilities reduce configuration complexity
- Regular updates and improvements to language models through cloud deployment
- Best choice for applications requiring broad multilingual support

**Tesseract** - Mature multilingual capabilities
- Supports 100+ languages through downloadable language packs
- Long development history has resulted in well-tested language models
- Performance varies significantly between languages, with better support for Latin-based scripts
- Requires manual language specification for optimal performance
- Community-driven language model development ensures ongoing support
- Good choice for applications with known target languages

**Surya** - Modern multilingual approach
- Supports 90+ languages with focus on modern transformer-based language understanding
- Designed with multilingual capabilities from the ground up
- Better performance on complex scripts and mixed-language documents
- Active development means rapid improvements in language support
- Good balance between language coverage and performance quality

**EasyOCR** - Accessible multilingual support
- Supports 80+ languages with emphasis on ease of use
- Simple language specification through parameter configuration
- Consistent API across different languages reduces implementation complexity
- Good performance on major languages but may struggle with less common scripts
- Regular updates add support for additional languages

**DocTR** - Limited but focused language support
- More limited language support compared to other models
- Focus on document-specific language understanding rather than broad coverage
- Strong performance on supported languages, particularly for document analysis
- May require additional configuration for non-Latin scripts
- Best suited for applications with specific language requirements

### 4. Integration and Deployment Analysis

**Easy Integration Models:**

**Google Vision API** - Simplest integration path
- RESTful API requires only HTTP requests, no local model installation
- Comprehensive documentation and client libraries for major programming languages
- No hardware requirements or dependency management on client side
- Authentication through Google Cloud credentials
- Immediate availability without setup time
- Ideal for rapid prototyping and production deployment with minimal infrastructure

**EasyOCR** - Straightforward Python integration
- Single pip install command with automatic dependency resolution
- Simple Python API with minimal configuration required
- Automatic model downloading on first use
- Good documentation and community examples
- Works out-of-the-box on most Python environments
- Suitable for Python-based applications requiring quick OCR integration

**Moderate Integration Complexity:**

**Tesseract** - System-level installation required
- Requires system package installation (apt-get, brew, etc.) beyond Python packages
- Language pack management for multilingual support
- Configuration file management for optimal performance
- Python wrapper (pytesseract) provides easier API access
- Well-documented but requires understanding of system dependencies
- Good choice when system-level control is acceptable

**Surya** - Modern Python package with dependencies
- Standard pip installation but with specific dependency requirements
- May require GPU drivers and CUDA installation for optimal performance
- Modern Python packaging standards make installation relatively straightforward
- Good documentation for setup and configuration
- Requires understanding of deep learning environment setup

**Complex Integration:**

**DocTR** - Comprehensive setup requirements
- Multiple dependency categories: PyTorch, computer vision libraries, document processing tools
- GPU setup strongly recommended for acceptable performance
- Potential version conflicts between dependencies
- Requires understanding of PyTorch ecosystem and CUDA configuration
- More complex model management and configuration options
- Best suited for teams with machine learning infrastructure experience

### 5. Cost and Economic Analysis

**Google Vision API** - Pay-per-use model
- Direct cost: $1.50 per 1000 requests after free tier (1000 requests/month)
- Hidden costs: Google Cloud Platform account setup, potential data egress charges
- Cost predictability: Linear scaling with usage, easy to budget for known volumes
- Economic advantages: No infrastructure costs, no maintenance overhead, immediate scalability
- Cost considerations: Can become expensive for high-volume applications (>100k requests/month)
- Break-even analysis: Cost-effective for low to medium volume applications or when development time is valuable
- Total cost of ownership: Low for small applications, potentially high for large-scale deployment

**Open Source Models** - Free software with infrastructure costs
- Direct cost: $0 for software licensing
- Infrastructure costs: Server hardware, GPU rental, electricity, maintenance
- Development costs: Integration time, model optimization, troubleshooting
- Operational costs: Monitoring, updates, scaling infrastructure
- Hidden costs: Developer time for setup, maintenance, and optimization
- Economic advantages: No per-request fees, full control over deployment, data privacy
- Long-term considerations: Infrastructure investment pays off for high-volume applications

### 6. License and Infrastructure Requirements Analysis

**Google Vision API**
- License: Proprietary service with terms of service restrictions
- Data privacy: Data processed on Google's servers, subject to Google's privacy policies
- Deployment: Cloud-only, no on-premises option
- Infrastructure: No local hardware requirements, internet connectivity essential
- Scalability: Automatic scaling handled by Google's infrastructure
- Compliance: May not meet strict data residency or air-gapped environment requirements
- Vendor lock-in: Dependency on Google's service availability and pricing

**Tesseract**
- License: Apache 2.0 - permissive open source license allowing commercial use
- Data privacy: Complete local processing, no data leaves your infrastructure
- Deployment: Flexible - local, cloud, or hybrid deployment options
- Infrastructure: CPU-based processing, minimal hardware requirements
- Scalability: Manual scaling through load balancing and horizontal scaling
- Compliance: Suitable for strict data privacy and air-gapped environments
- Independence: No vendor dependencies, full control over deployment and updates

**EasyOCR**
- License: Apache 2.0 - open source with commercial use permissions
- Data privacy: Local processing ensures data privacy and security
- Deployment: Local or cloud deployment with containerization support
- Infrastructure: GPU recommended but not required, CPU fallback available
- Scalability: Good scalability with proper infrastructure planning
- Compliance: Meets most data privacy requirements through local processing
- Community: Active community support and regular updates

**DocTR**
- License: Apache 2.0 - permissive license suitable for commercial applications
- Data privacy: Complete local control over data processing and storage
- Deployment: Requires more sophisticated infrastructure due to PyTorch dependencies
- Infrastructure: GPU strongly recommended for production use, higher memory requirements
- Scalability: Excellent scalability with proper GPU infrastructure
- Compliance: Ideal for environments requiring complete data control
- Maintenance: Requires more technical expertise for deployment and maintenance

**Surya**
- License: Apache 2.0 - open source with flexible usage terms
- Data privacy: Local processing maintains data privacy and security
- Deployment: Modern containerized deployment options available
- Infrastructure: GPU recommended for optimal performance, moderate resource requirements
- Scalability: Good scalability potential with modern architecture
- Compliance: Suitable for privacy-conscious applications
- Innovation: Cutting-edge technology with active development and improvements

## Comprehensive Recommendations

### For Maximum Accuracy and Minimal Setup Time
**Google Vision API** is the optimal choice when:
- Accuracy is the primary concern and budget allows for per-request pricing
- Development time is limited and immediate deployment is required
- Internet connectivity is reliable and data privacy concerns are minimal
- Processing volume is low to medium (under 50k requests/month)
- Team lacks machine learning infrastructure expertise

### For Best Free Alternative with Document Analysis
**DocTR** is recommended when:
- Budget constraints prevent use of paid services
- Document layout analysis and spatial understanding are important
- Team has machine learning infrastructure capabilities
- Processing volume justifies infrastructure investment
- Data privacy and local processing are requirements
- Accuracy requirements are high but Google Vision API costs are prohibitive

### For Real-time and Resource-Constrained Applications
**Tesseract** is ideal when:
- Fast, local processing is essential for application performance
- Hardware resources are limited (CPU-only environments)
- Offline operation is required (no internet connectivity)
- Simple text extraction without complex layout analysis is sufficient
- Long-term stability and mature technology are valued
- Integration with existing systems requires minimal dependencies

### For Modern Applications with Balanced Requirements
**Surya** is suitable when:
- Modern OCR technology with good multilingual support is needed
- Balance between accuracy, speed, and resource requirements is important
- Team prefers cutting-edge open source solutions
- Document processing requirements are moderate to complex
- GPU resources are available but not unlimited
- Active development and community support are valued

### For Quick Prototyping and Simple Applications
**EasyOCR** is appropriate when:
- Rapid prototyping and proof-of-concept development is the goal
- Simplicity of integration outweighs maximum accuracy requirements
- Team has limited OCR experience and needs accessible tools
- Processing requirements are basic text extraction
- Budget is constrained but some accuracy trade-offs are acceptable
- Python-based development environment is preferred

## Installation & Usage

### Requirements
```bash
pip install easyocr pytesseract google-cloud-vision
pip install python-doctr[torch]
pip install surya-ocr
```

### Usage Example
```python
# Initialize models
from complete_ocr_evaluation import initialize_ocr_models, run_single_ocr_evaluation

# Load your samples
samples = load_your_samples()

# Run evaluation
results = run_single_ocr_evaluation(samples, 'google_vision')
```

## Project Structure

```
├── README.md
├── OCR_MODELS_COMPARISON.ipynb    # Main Jupyter notebook
├── complete_ocr_evaluation.py     # Python script version
├── model_comparison_summary.csv   # Results summary
└── datasets/
    ├── English_OCR_dataset/
    └── French_OCR_dataset/
```

