# 📚 Fine-tuning NMT with PDF Document Alignment

A complete end-to-end Neural Machine Translation fine-tuning project that extracts parallel text from bilingual PDFs, builds a high-quality aligned corpus, and fine-tunes a MarianMT model for domain-specific translation.

## 🎯 Project Overview

This project demonstrates how to create a custom parallel corpus from bilingual PDF documents and use it to fine-tune a pre-trained translation model. Using Tony Attwood's book on Asperger's Syndrome (English-French editions) as a case study, the workflow covers the entire pipeline from PDF extraction to model evaluation.

**Use Case**: Literary/technical document translation with domain-specific terminology  
**Language Pair**: English → French  
**Base Model**: MarianMT (Helsinki-NLP)

### 📖 Source Material Citation

**English Original:**
- **Title**: *The Complete Guide to Asperger's Syndrome*
- **Author**: Tony Attwood
- **Publisher**: Jessica Kingsley Publishers, London and Philadelphia

**French Translation:**
- **Title**: *Le syndrome d'Asperger - Guide complet* (3e édition)
- **Author**: Tony Attwood
- **Translation**: Josef Schovanec
- **Scientific Revision**: Elaine Hardiman-Taveau and Cécile Veasna Malterre
- **Publisher**: Groupe De Boeck s.a., Brussels, Belgium (2010)

> **Note**: These copyrighted works are used solely for educational purposes in machine translation research. The source PDFs are not included in this repository. Users must obtain legal copies to reproduce this work.

## 🗂️ Project Structure

```
NMT with PDF Document Alignment/
├── pdf_books/                          # Source PDFs (not in repo - see .gitignore)
│   ├── Attwood_Asperger_EN.pdf
│   └── Attwood_Asperger_FR.pdf
├── data/                               # Generated data (git-ignored except samples)
│   ├── extracted/                      # Chapter-level text extraction
│   ├── aligned/                        # Sentence-aligned pairs
│   └── processed/                      # Train/val/test splits
├── 01_pdf_extraction_and_exploration.ipynb
├── 02_text_alignment_and_corpus_building.ipynb
├── 03_model_finetuning_and_evaluation_from_colab.ipynb
└── README.md
```

## 📓 Notebooks

### **1. PDF Extraction and Exploration** (`01_pdf_extraction_and_exploration.ipynb`)

Extracts and preprocesses text from bilingual PDF documents.

**What it does:**
- Loads English and French PDF versions of the same book
- Extracts raw text with chapter segmentation
- Identifies structural alignment between versions
- Cleans and validates extracted content
- Exports chapter-level JSON files

**Key Libraries**: `PyMuPDF` (fitz), `pathlib`, `pandas`, `matplotlib`  
**Runtime**: ~5-10 minutes  
**Environment**: Local (VS Code)

**Outputs:**
- `data/extracted/chapters_en.json`
- `data/extracted/chapters_fr.json`
- `data/extracted/metadata.json`

---

### **2. Text Alignment and Corpus Building** (`02_text_alignment_and_corpus_building.ipynb`)

Creates a high-quality parallel corpus through advanced sentence alignment.

**What it does:**
- Sentence segmentation using spaCy (EN/FR)
- Multiple alignment strategies:
  - Length-based (Gale-Church algorithm)
  - Semantic similarity (sentence embeddings)
  - Hybrid approach combining both
- Quality filtering and validation
- Train/validation/test split (80/10/10)

**Key Libraries**: `spacy`, `sentence-transformers`, `pandas`, `numpy`  
**Runtime**: ~15-30 minutes  
**Environment**: Local (VS Code)

**Outputs:**
- `data/aligned/parallel_corpus.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

---

### **3. Model Fine-tuning and Evaluation** (`03_model_finetuning_and_evaluation_from_colab.ipynb`)

Fine-tunes MarianMT on the custom corpus and evaluates performance.

**What it does:**
- Loads processed parallel corpus
- Prepares data for Hugging Face Transformers
- Fine-tunes pretrained MarianMT (EN→FR)
- Evaluates with BLEU, chrF metrics
- Compares baseline vs. fine-tuned performance
- Saves model for deployment

**Key Libraries**: `transformers`, `datasets`, `torch`, `evaluate`  
**Platform**: **Google Colab (GPU recommended)**  
**Runtime**: ~1-2 hours (depends on GPU)

> **⚠️ Note**: This notebook is designed for Google Colab and may not render properly on GitHub. For best viewing and execution:  
> 📓 [**Open in Colab**](https://colab.research.google.com/drive/1cQJe0Z9MKQDxaxnJXD9JiPrI9azrHbUU#scrollTo=65b513b6)

**Outputs:**
- `models/marian-finetuned-asperger-en-fr/` (fine-tuned model)
- `results/evaluation_metrics.json`
- `results/sample_translations.txt`

## 🛠️ Requirements

### Local Environment (Notebooks 1 & 2)

```bash
# PDF processing
pip install PyMuPDF

# NLP and alignment
pip install spacy sentence-transformers

# Spacy language models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

# Data processing and visualization
pip install pandas numpy matplotlib

# Jupyter
pip install jupyter notebook
```

### Google Colab (Notebook 3)

All required libraries are pre-installed or installed via notebook cells. Requires GPU runtime (T4 minimum, V100/A100 recommended for faster training).

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AgaHei/NMT_Fine_Tuning.git
cd NMT_Fine_Tuning
```

### 2. Prepare Your Data

**Option A**: Use your own bilingual PDFs
- Place English PDF in `pdf_books/[name]_EN.pdf`
- Place French PDF in `pdf_books/[name]_FR.pdf`
- Update file paths in Notebook 1

**Option B**: Use the Attwood example
- Source PDFs are not included in the repository (copyright/size reasons)
- If you have access to the books, place them in `pdf_books/`

### 3. Run the Pipeline

**Step 1**: PDF Extraction (Local)
```bash
jupyter notebook 01_pdf_extraction_and_exploration.ipynb
```
Run all cells to extract and segment chapters.

**Step 2**: Text Alignment (Local)
```bash
jupyter notebook 02_text_alignment_and_corpus_building.ipynb
```
Run all cells to create aligned corpus and data splits.

**Step 3**: Model Fine-tuning (Colab)
1. Upload Notebook 3 to Google Colab
2. Upload `data/processed/` folder contents to Colab
3. Set runtime to GPU (Runtime → Change runtime type → T4)
4. Run all cells

### 4. Access Fine-tuned Model

After Notebook 3 completes:
- Download model from Colab: `models/marian-finetuned-asperger-en-fr/`
- Or upload directly to Hugging Face Hub (code included in notebook)

## 📊 Key Features

✅ **End-to-end pipeline**: From PDF to fine-tuned model  
✅ **Multiple alignment strategies**: Length-based + semantic  
✅ **Quality filtering**: Ensures high-quality parallel data  
✅ **Comprehensive evaluation**: BLEU, chrF, human inspection  
✅ **Production-ready**: Exportable model for deployment  
✅ **Well-documented**: Clear explanations and best practices

## 🧪 Example Use Cases

This pipeline can be adapted for:
- **Literary translation**: Books, novels, essays
- **Technical documentation**: Manuals, guides, specifications
- **Academic papers**: Research articles, theses
- **Legal documents**: Contracts, agreements (with proper preprocessing)
- **Any bilingual PDF corpus**: Medical, financial, educational content

## 📈 Expected Results

Based on the Attwood Asperger's book corpus:
- **Corpus size**: ~2,000-5,000 aligned sentence pairs (varies by book)
- **BLEU improvement**: +5-15 points over baseline (domain-specific)
- **chrF improvement**: More robust on literary text
- **Training time**: 1-2 hours on T4 GPU

## 🎓 Learning Outcomes

This project teaches:
- PDF text extraction and preprocessing
- Sentence alignment algorithms (Gale-Church, embedding-based)
- Parallel corpus creation and quality control
- Fine-tuning transformer models with Hugging Face
- NMT evaluation metrics and interpretation
- End-to-end ML pipeline design

## ⚠️ Important Notes

### Source Material & Copyright
This project uses copyrighted bilingual texts for educational purposes in machine translation research. Full citations are provided in the "Project Overview" section above.

**Important:**
- The source PDFs are **not included** in this repository
- Users must obtain **legal copies** of the books to reproduce this work
- This project demonstrates the **technical pipeline** for NMT fine-tuning
- The extracted text and aligned corpus are **not distributed**
- Any use of the source material must comply with copyright laws

### Data Not Included
The source PDF files and generated data folders are not included in this repository due to:
- **Copyright**: The books are copyrighted material by their respective publishers
- **File size**: PDFs and generated data can be large
- **Privacy**: You may use your own proprietary documents

See `.gitignore` for excluded files.

### Reproducibility
To reproduce the exact results:
1. Obtain the same source PDFs
2. Use the same random seeds (specified in notebooks)
3. Use similar GPU hardware for training

### Customization
The notebooks are designed to be easily adaptable:
- Modify file paths for your own PDFs
- Adjust alignment thresholds for your data quality
- Change hyperparameters for different corpus sizes
- Adapt to other language pairs (requires different spaCy models)

## 🤝 Contributing

This is a learning project, but suggestions are welcome! Feel free to:
- Report issues or bugs
- Suggest improvements to the pipeline
- Share your own fine-tuning results
- Propose new features or use cases

## 📄 License

This project is open source for educational purposes. Note that:
- The code and pipeline are freely available
- Source PDFs are subject to their respective copyrights
- Fine-tuned models inherit the license of the base model (MarianMT)

## 🔗 Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [MarianMT Models](https://huggingface.co/Helsinki-NLP)
- [Sentence Transformers](https://www.sbert.net/)
- [spaCy Documentation](https://spacy.io/)
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- [BLEU Score Explanation](https://en.wikipedia.org/wiki/BLEU)

## 📬 Contact

**Author**: Learning NMT Journey  
**GitHub**: [@AgaHei](https://github.com/AgaHei)  
**Project Repository**: [NMT_Fine_Tuning](https://github.com/AgaHei/NMT_Fine_Tuning)

---

**Last Updated**: November 2025  
**Status**: ✅ Complete and functional

*Part of a self-directed learning series on Neural Machine Translation. See also: [Learning_NMT](https://github.com/AgaHei/Learning_NMT) for foundational concepts.*
