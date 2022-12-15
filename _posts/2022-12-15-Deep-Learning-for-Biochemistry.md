---
layout: post
title: Deep Learning for Biochemistry
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Introduction to Deep Learning for Precision Medicine, Genomics, Protein Folding, Computational Chemistry. Biomedicine, Virus Identification.

---
## [Deep Learning for Precision Medicine](https://www.nature.com/articles/s41746-019-0191-0)

* Historical milestones related to precision medicine and artificial intelligence.<br>

![](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-019-0191-0/MediaObjects/41746_2019_191_Fig2_HTML.png?as=webp)

---
* Complex unresolved problems in neurodevelopmental disorders that artificial intelligence algorithms can create an impact<br>

![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41746-019-0191-0/MediaObjects/41746_2019_191_Fig3_HTML.png?as=webp)

---
## [Deep Learning for Genomics](https://codete.com/blog/machine-learning-genomics)
* Gene Editing
* Genome Sequencing
* Clinical workflows
* Consumer genomics products
* Pharmacy genomics
* Genetic screening of newborns
* Agriculture

[Artificial intelligence in clinical and genomic diagnostics](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-019-0689-8)<br>

![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs13073-019-0689-8/MediaObjects/13073_2019_689_Fig1_HTML.png?as=webp)

---
### Deep Learning for GWAS
* [Deep Learning Classification of Polygenic Obesity using Genome Wide Association Study SNPs (多基因肥胖分類 using GWAS)](https://arxiv.org/ftp/arxiv/papers/1804/1804.03198.pdf)

* [Using the structure of genome data in the design of deep neural networks for predicting Amyotrophic Lateral Sclerosis from genotype (以基因型預測肌萎縮性側索硬化)](https://ir.cwi.nl/pub/28785/28785.pdf)
  - Code: [ALS-Deeplearning](https://github.com/byin-cwi/ALS-Deeplearning)

---
## Deep Learning in Biomedicine
Course: [Deep Learning in Genomics and Biomedicine](https://canvas.stanford.edu/courses/70852)<br>

* [DanQ: A Hybrid Convolutional and Recurrent Deep Neural Network for Quantifying the Function of DNA Sequences](https://pubmed.ncbi.nlm.nih.gov/27084946/)
  - Code: [http://github.com/uci-cbcl/DanQ](http://github.com/uci-cbcl/DanQ)
  
* [Basenji: Sequential regulatory activity prediction across chromosomes with convolutional neural networks](https://www.biorxiv.org/content/10.1101/161851v1)
  - Code: [https://github.com/calico/basenji](https://github.com/calico/basenji)

* [FIDDLE: An integrative deep learning framework for functional genomic data inference](https://www.biorxiv.org/content/10.1101/081380v1)
  - Code: [https://github.com/ueser/FIDDLE](https://github.com/ueser/FIDDLE)

---
### Biopython
![](https://biopython.org/assets/images/biopython_logo_s.png)
`pip3 install biopython`<br>

* [Biopython Tutorial and Cookbook](http://biopython.org/DIST/docs/tutorial/Tutorial.html)<br>

---
## Genome Basics

### [Differences Between DNA and RNA](https://byjus.com/biology/difference-between-dna-and-rna/)
![](https://cdn1.byjus.com/wp-content/uploads/2017/11/Difference-Between-DNA-and-RNA.png)
[DNA vs. RNA – 5 Key Differences and Comparison](https://www.technologynetworks.com/genomics/lists/what-are-the-key-differences-between-dna-and-rna-296719)

---
### Genome, Transcriptome, Proteome, Metabolome
* Genome (基因組)
* Transcriptome (轉錄組)
* Proteome (蛋白質組)
* Metabolome (代謝組)

![](https://i.ytimg.com/vi/FJ5iN-v0MFM/maxresdefault.jpg)

---
### RNA-Seq (核糖核酸測序)
RNA-seq (核糖核酸測序)也被稱為Whole Transcriptome Shotgun Sequencing (全轉錄物組散彈槍法測序)是基於Next Generation Sequencing(第二代測序技術)的轉錄組學研究方法<br>
![](https://www.researchgate.net/profile/Nona-Farbehi/publication/332280191/figure/fig1/AS:806612591734787@1569322836102/Single-cell-RNAseq-sample-journey-Schematic-representation-of-a-typical-workflow-for-an.ppm)

---
## Deep DNA sequence analysis

### Basset
Train deep convolutional neural networks to learn highly accurate models of DNA sequence activity such as accessibility (via **DNaseI-seq** or **ATAC-seq**), protein binding (via **ChIP-seq**), and chromatin state.<br>

* [Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4937568/)<br>
  - Code: [https://github.com/davek44/Basset](https://github.com/davek44/Basset)<br>
<br>
* [Deep learning models predict regulatory variants in pancreatic islets and refine type 2 diabetes association signals](https://elifesciences.org/articles/51503)
  - Code: [https://github.com/agawes/islet_CNN](https://github.com/agawes/islet_CNN)<br>

---
### [ENCODE Project Common Cell Types](https://www.genome.gov/encode-project-common-cell-types)
The Encyclopedia of DNA Elements (ENCODE) Project seeks to identify functional elements in the human genome.<br>
* Tier 1:
  - **GM12878:**  is a lymphoblastoid cell line (淋巴母細胞系)
  - **K562** is an immortalized cell line (永生細胞系). It is a widely used model for cell biology, biochemistry, and erythropoiesis (紅血球細胞生成)
  - **H1** human embryonic stem cells
* Tier 2:
  - **HeLa-S3** is an immortalized cell line that was derived from a cervical cancer (宮頸癌) patient. 
  - **HepG2** is a cell line derived from a male patient with liver carcinoma (肝癌).
  - **HUVEC** (human umbilical vein endothelial cells) (人臍靜脈內皮細胞)
* Tier 2.5
  - SK-N-SH, IMR90 (ATCC CCL-186), A549 (ATCC CCL-185), MCF7 (ATCC HTB-22), HMEC or LHCM, CD14+, CD20+, Primary heart or liver cells, Differentiated H1 cells

---
### DeepCTCFLoop
Code: [https://github.com/BioDataLearning/DeepCTCFLoop](https://github.com/BioDataLearning/DeepCTCFLoop)<br>
DeepCTCFLoop is a deep learning model to predict whether a chromatin loop can be formed between a pair of convergent or tandem CTCF motifs<br>
DeepCTCFLoop was evaluated on three different cell types **GM12878**, **Hela** and **K562**<br>
* Training
  - `python3 train.py -f Data/GM12878_pos_seq.fasta -n Data/GM12878_neg_seq.fasta -o GM12878.output`<br>
* Motif Visualization
  - `python3 get_motifs.py -f  Data/GM12878_pos_seq.fasta -n Data/GM12878_neg_seq.fasta`<br>

---
### DARTS
**Blog:** [邢毅團隊利用深度學習強化RNA可變剪接分析的準確性](https://kknews.cc/science/5n86xgl.html)<br>
**Paper:** [Deep-learning Augmented RNA-seq analysis of Transcript Splicing](https://pubmed.ncbi.nlm.nih.gov/30923373/)<br>
![](http://i2.kknews.cc/l9emUOkuBwzsAiAr_szoLJmUT8JWB_g/0.jpg)
**Code:** [https://github.com/Xinglab/DARTS](https://github.com/Xinglab/DARTS)<br>

---
### Coda 
* Coda: a convolutional denoising algorithm for genome-wide ChIP-seq data<br>
 - ChIP-sequencing is a method used to analyze protein interactions with DNA. <br>
 - ChIP-seq combines [chromatin immunoprecipitation 染色質免疫沉澱](https://en.wikipedia.org/wiki/Chromatin_immunoprecipitation) (ChIP) with massively parallel DNA sequencing to identify the binding sites of DNA-associated proteins.<br>

* **Paper:** [Denoising genome-wide histone ChIP-seq with CNN](https://www.biorxiv.org/content/10.1101/052118v2.full)
![](https://www.biorxiv.org/content/biorxiv/early/2017/01/27/052118/F2.medium.gif)

* **Code:** [https://github.com/kundajelab/coda](https://github.com/kundajelab/coda)

---
### SNP (Single Nucleotide Polymorphism) 單核苷酸多型性
<iframe width="620" height="349" src="https://www.youtube.com/embed/mVhrWNRUeW0" title="SNPs   Single Nucleotide Polymorphism Better Explained 1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

[SNP(單核苷酸多型性)](http://bioinfo.cs.ccu.edu.tw/wiki/doku.php?id=single_nucleotide_polymorphism_snp): DNA序列中的單一鹼基對(base pair)變異，一般指變異頻率大於1%的單核苷酸變異。<br>

![](http://bioinfo.cs.ccu.edu.tw/wiki/lib/exe/fetch.php?w=&h=&cache=cache&media=416px-dna-snp.svg.png)
* 於所有可能的DNA序列差異性中，SNP是最普遍發生的一種遺傳變異。在人體中，SNP的發生機率大約是0.1%，也就是每1200至1500個鹼基對中，就可能有一個SNP。
* 目前科學界已發現了約400萬個SNPs。平均而言，每1kb長的DNA中，就有一個SNP存在；換言之每個人的DNA序列中，每隔1kb單位長度，就至少會發生一個「單一鹼基變異」。由於SNP的發生頻率非常之高，故SNP常被當作一種基因標記(genetic marker)，已用來進行研究。

---
### DeepCpG
**Paper:** [DeepCpG: accurate prediction of single-cell DNA methylation states using deep learning](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1189-z)<br>
**Code:** [https://github.com/cangermueller/deepcpg](https://github.com/cangermueller/deepcpg)<br>
![](https://github.com/cangermueller/deepcpg/raw/master/docs/source/fig1.png)

---
### DeepTSS (Transcription Start Site)
**Paper:** [Genome Functional Annotation across Species using Deep CNN](https://www.biorxiv.org/content/10.1101/330308v4.full)<br>
![](https://www.biorxiv.org/content/biorxiv/early/2019/06/07/330308/F5.large.jpg?width=800&height=600&carousel=1)
**Code:** [https://github.com/StudyTSS/DeepTSS](https://github.com/StudyTSS/DeepTSS)<br>
**Dataset:** The TSS positions are collected from the reference genomes for human (hg38) and mouse (mm10) species. [http://hgdownload.soe.ucsc.edu/](http://hgdownload.soe.ucsc.edu/)<br>
TSS positions over the entire human and mouse genomes data [http://egg.wustl.edu/](http://egg.wustl.edu/), the gene annotation is taken from RefGene<br>

---
### DeepFunNet
**Paper:** [DeepFunNet: Deep Learning for Gene Functional Similarity Network Construction](https://www.semanticscholar.org/paper/DeepFunNet%3A-Deep-Learning-for-Gene-Functional-Ahn-Son/8bd66a65f5f7a9caf33ce2456832ca7ecefdc1a4)<br>
![](https://d3i71xaburhd42.cloudfront.net/8bd66a65f5f7a9caf33ce2456832ca7ecefdc1a4/4-Figure2-1.png)
![](https://d3i71xaburhd42.cloudfront.net/8bd66a65f5f7a9caf33ce2456832ca7ecefdc1a4/4-Figure3-1.png)
* http://geneontology.org/docs/ontology-documentation/

---
### Population Genetic Inference
**Paper:** [The Unreasonable Effectiveness of Convolutional Neural Networks in Population Genetic Inference](https://www.biorxiv.org/content/10.1101/336073v3.full)<br>
![](https://www.biorxiv.org/content/biorxiv/early/2018/11/27/336073/F1.large.jpg?width=800&height=600&carousel=1)
**Code:** [https://github.com/flag0010/pop_gen_cnn](https://github.com/flag0010/pop_gen_cnn)<br>

---
### GANs for Biological Image Synthesis
**Paper:** [GANs for Biological Image Synthesis](https://arxiv.org/abs/1708.04692)<br>
![](https://media.arxiv-vanity.com/render-output/5891906/x1.png)
**Code:** [https://github.com/aosokin/biogans](https://github.com/aosokin/biogans)<br>
**Code:** [https://github.com/VladSkripniuk/gans](https://github.com/VladSkripniuk/gans)<br>
**Dataset:** LIN dataset<br>
LIN dataset contains photographs of 41 proteins in fission yeast cells.<br>

---
### DeepGP
Genomic Selection is the breeding strategy consisting in predicting complex traits using genomic-wide genetic markers and it is standard in many animal and plant breeding schemes.<br>
**Paper:** [A Guide on Deep Learning for Complex Trait Genomic]()<br>
**Code:** [DLpipeine](https://github.com/miguelperezenciso/Dlpipeline)<br>
**Code:** [DeepGP](https://github.com/lauzingaretti/DeepGP)<br>
The DeepGP package implements Multilayer Perceptron Networks (MLP), Convolutional Neural Network (CNN), Ridge Regression and Lasso Regression to Genomic Prediction purposes.<br>

---
## Biochemistry Tools

### [PubChem Sketcher V2.4](https://pubchem.ncbi.nlm.nih.gov/edit3/index.html)
![](https://www.researchgate.net/publication/42344113/figure/fig14/AS:667200293724169@1536084354467/The-PubChem-Chemical-Structure-Sketcher.png)

---
### [Molview](http://molview.org/)
![](https://i.ytimg.com/vi/QIF6amwfIGE/maxresdefault.jpg)
<iframe width="895" height="503" src="https://www.youtube.com/embed/iLNYkHZks8g" title="How to use MolView to draw molecules" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Protein Folding

### Attention Based Protein Structure Prediction
**Kaggle**: [https://www.kaggle.com/code/basu369victor/attention-based-protein-structure-prediction](https://www.kaggle.com/code/basu369victor/attention-based-protein-structure-prediction)<br>
![](https://raw.githubusercontent.com/lucidrains/alphafold2/main/images/alphafold2.png)

---
### AlphaFold 2
**Blog:** [AlphaFold reveals the structure of the protein universe](https://www.deepmind.com/blog/alphafold-reveals-the-structure-of-the-protein-universe)<br>

**Paper:** [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)<br>
![](https://www.researchgate.net/publication/353275939/figure/fig1/AS:1058388020580354@1629350776780/AlphaFold-produces-highly-accurate-structures-a-The-performance-of-AlphaFold-on-the.png)
![](https://www.researchgate.net/publication/353275939/figure/fig3/AS:1058388024778752@1629350777146/Architectural-details-a-Evoformer-block-Arrows-show-the-information-flow-The-shape-of.png)

**Blog:** [DeepMind's AlphaFold 2 reveal: Convolutions are out, attention is in](https://www.zdnet.com/article/deepminds-alphafold-2-reveal-what-we-learned-and-didnt-learn/)<br>

**Code:** [https://github.com/deepmind/alphafold](https://github.com/deepmind/alphafold)<br>
- [AlphaFold.ipynb](https://github.com/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)

---
### [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)
AlphaFold DB provides open access to over 200 million protein structure predictions to accelerate scientific research.<br>
![](https://assets-global.website-files.com/621e749a546b7592125f38ed/62e144dd63664354fa10ac1f_AF_Announcement_Vis1%20(2).svg)

* **Q8W3K0**: A potential plant disease resistance protein. Mean pLDDT 82.24.<br>
![](https://alphafold.ebi.ac.uk/assets/img/Q8W3K0.png)

<iframe width="655" height="368" src="https://www.youtube.com/embed/Gk-PyJSNlcI" title="AlphaFold Protein Structure Database" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Deep Learning for Computational Chemistry

### OpenChem
<img width="25%" height="25%" src="https://github.com/Mariewelt/OpenChem/blob/master/docs/logo.png?raw=true">
OpenChem is a deep learning toolkit for Computational Chemistry with PyTorch backend.<br>
**Code:** [https://github.com/Mariewelt/OpenChem](https://github.com/Mariewelt/OpenChem)<br>
  - [Documentation](https://mariewelt.github.io/OpenChem/html/index.html)<br>
  - [Tutorials and Recipes](https://mariewelt.github.io/OpenChem/html/tutorials/blocks.html#)<br>

---
### Organic Chemistry Reaction Prediction
**Paper:** [Predicting Outcomes of Complex Organic Chemistry Reactions using Neural Sequence-to-Sequence Models](https://arxiv.org/abs/1711.04810)<br>
![](https://pubs.rsc.org/en/Content/Image/GA/C8SC02339E)
**Code:** [Organic Chemistry Reaction Prediction using NMT with Attention](https://github.com/ManzoorElahi/organic-chemistry-reaction-prediction-using-NMT)<br>
The model in version 2 is slightly based on the model discussed in [Asynchronous Bidirectional Decoding for Neural Machine Translation](https://arxiv.org/abs/1801.05122).

### Retrosynthesis Planner
**Paper:** [Planning chemical syntheses with deep neural networks and symbolic AI](https://arxiv.org/pdf/1708.04202.pdf)<br>
![](https://2.bp.blogspot.com/-Wmpv30Udq8s/Wr4nqsJJPpI/AAAAAAAAEoM/ebT2YUiXUCEEbF7uglsi61g4KX3owtQ7ACLcBGAs/s640/screenshot_1142.png)
**Slides:** [CSC2547_learning_to_plan_chemical_synthesis.pdf](https://duvenaud.github.io/learning-to-search/slides/week3/CSC2547_learning_to_plan_chemical_synthesis.pdf)<br>
**Code:** [https://github.com/frnsys/retrosynthesis_planner](https://github.com/frnsys/retrosynthesis_planner)<br>

---
### Step-wise Chemical Synthesis prediction
**Code:** [A GGNN-GWM based step-wise framework for Chemical Synthesis Prediction](https://github.com/pfnet-research/step-wise-chemical-synthesis-prediction)<br>
![](https://github.com/pfnet-research/step-wise-chemical-synthesis-prediction/raw/master/figure/framework.png)
![](https://github.com/pfnet-research/step-wise-chemical-synthesis-prediction/raw/master/figure/example2.png)

---
### Retrosynthesis 
**Paper:** [Decomposing Retrosynthesis into Reactive Center Prediction and Molecule Generation](https://www.biorxiv.org/content/10.1101/677849v1.full)<br>
<img width="50%" height="50%" src="https://www.biorxiv.org/content/biorxiv/early/2019/06/21/677849/F1.large.jpg">
<img width="50%" height="50%" src="https://www.biorxiv.org/content/biorxiv/early/2019/06/21/677849/F3.large.jpg">

---
### RetroXpert
**Paper:** [RetroXpert: Decompose Retrosynthesis Prediction like a Chemist](https://arxiv.org/abs/2011.02893)<br>
![](https://d3i71xaburhd42.cloudfront.net/6ff971e6036cbffdd819e34c17293f724028a5fd/4-Figure1-1.png)
**Code:** [https://github.com/uta-smile/RetroXpert](https://github.com/uta-smile/RetroXpert)<br>

---
### Neural Message Passing for Quantum Chemistry
**Paper:** [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)<br>
<img width="50%" height="50%" src="https://production-media.paperswithcode.com/methods/MPNN_afcPv22.png">
A Message Passing Neural Network predicts quantum properties of an organic molecule by modeling a computationally expensive DFT calculation<br>

**Code:** [https://github.com/priba/nmp_qc](https://github.com/priba/nmp_qc)<br>

---
## Biomedicine

### DeepChem
**Paper:** [Low Data Drug Discovery with One-Shot Learning](https://pubs.acs.org/doi/pdf/10.1021/acscentsci.6b00367)<br>
![](https://pubs.acs.org/cms/10.1021/acscentsci.6b00367/asset/images/medium/oc-2016-00367d_0001.gif)
**Code:** [https://github.com/deepchem/deepchem](https://github.com/deepchem/deepchem)<br>
* Tutorials:
  - [Modeling_Protein_Ligand_Interactions.ipynb](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/13_Modeling_Protein_Ligand_Interactions.ipynb)
  - [Predicting_Ki_of_Ligands_to_a_Protein.ipynb](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/12_Predicting_Ki_of_Ligands_to_a_Protein.ipynb)

---
### druGAN
**Paper**: [druGAN](https://pubs.acs.org/doi/pdf/10.1021/acs.molpharmaceut.7b00346)<br>
**Code:** [Gananath/DrugAI](https://github.com/Gananath/DrugAI)<br>
**Code:** [kumar1202/Drug-Discovery-using-GANs](https://github.com/kumar1202/Drug-Discovery-using-GANs)<br>
![](https://github.com/kumar1202/Drug-Discovery-using-GANs/raw/master/data/gan_RL.png)

---
### MoleculeNet
**Paper:** [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564)<br>
Datasets: In most datasets, **SMILES** strings are used to represent input molecules<br>
* **QM7/QM7b** datasets are subsets of the GDB-13 database, a database of nearly 1 billion stable and synthetically accessible organic molecules
* **QM8** dataset comes from a recent study on modeling quantum mechanical calculations of electronic spectra and excited state energy of small molecules
* **QM9** is a comprehensive dataset that provides geometric, energetic, electronic and thermodynamic properties for a subset of GDB-17 database
* **ESOL** is a small dataset consisting of water solubility data for 1128 compounds
* **FreeSolv** provides experimental and calculated hydration free energy of small molecules in water.
Lipophilicity is an important feature of drug molecules that affects both membrane permeability and solubility. This dataset, curated from ChEMBL database, provides experimental results of octanol/water distribution coefficient (logD at pH 7.4) of 4200 compounds
* **PCBA** is a database consisting of biological activities of small molecules generated by high-throughput screening
* **MUV** group is another benchmark dataset selected from PubChem BioAssay by applying a refined nearest neighbor analysis, contains 17 challenging tasks for around 90 thousand compounds
* **HIV** dataset was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for over 40,000 compounds
* **Tox21** contains qualitative toxicity measurements for 8014 compounds on 12 different targets, including nuclear receptors and stress response pathways
* **SIDER** is a database of marketed drugs and adverse drug reactions (ADR)
* **ClinTox** compares drugs approved by the FDA and drugs that have failed clinical trials for toxicity reasons

---
### [TDC Datasets](https://zitniklab.hms.harvard.edu/TDC/overview/)
<img width="50%" height="50%" src="https://tdcommons.ai/img/tdc_overview2.png">
* To install PyTDC
```
pip3 install PyTDC
```

* To obtain the dataset:
```
from tdc.Z import Y
data = Y(name = ‘X’)
splits = data.split()
```

* To obtain the Caco2 dataset from ADME therapeutic task in the single-instance prediction problem:
```
from tdc.single_pred import ADME
data = ADME(name = 'Caco2_Wang’) 
df = data.get_data() 
splits = data.get_split() 
```

---  
### 新型抗生素開發
**Blog:** [新型抗生素開發，機器學習立大功](https://highscope.ch.ntu.edu.tw/wordpress/?p=83888&fbclid=IwAR27wnnTtPmuSOs8nm79V74Kjn9VUqNrBMS8xqCyHnBlTPmyocnnL6Dj760)<br>
* 消息傳遞神經網路<br>
![](https://highscope.ch.ntu.edu.tw/wordpress/wp-content/uploads/2020/05/newantibiotic-1.png)
(圖片來源：M. Abdughani et al., 2019.)<br>

<u>References:</u><br>
1. C. Ross, [“Aided by machine learning, scientists find a novel antibiotic able to kill superbugs in mice”](https://www.statnews.com/2020/02/20/machine-learning-finds-novel-antibiotic-able-to-kill-superbugs/), STAT, 2020<br>
2. J. Gilmer et al., “Neural Message Passing for Quantum Chemistry”, arXiv.org, 2017
3. G. Dahl et al., “Predicting Properties of Molecules with Machine Learning”, Google AI blog, 2017
4. M. Abdughani et al., “Probing stop pair production at the LHC with graph neural networks”, Springer, 2019

---
## [Deep Learning in Proteomics](https://github.com/bzhanglab/deep_learning_in_proteomics/blob/master/README.md)
**Paper:** [Deep Learning in Proteomics](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.201900335)<br>
![](https://analyticalsciencejournals.onlinelibrary.wiley.com/cms/asset/f2f8b410-e7b1-40f8-b0fd-a90a2c200346/pmic13344-fig-0001-m.png)
![](https://analyticalsciencejournals.onlinelibrary.wiley.com/cms/asset/2aa81f0c-0d92-4315-b30a-7250597457fd/pmic13344-fig-0004-m.png)

---
### Peptide MS/MS spectrum prediction

1. **pDeep3**
  - Reference:
        - Zhou, Xie-Xuan, et al. "[pDeep: predicting MS/MS spectra of peptides with deep learning](https://pubs.acs.org/doi/10.1021/acs.analchem.7b02566)." *Analytical chemistry* 89.23 (2017): 12690-12697.  
        - Zeng, Wen-Feng, et al. "[MS/MS spectrum prediction for modified peptides using pDeep2 trained by transfer learning](https://pubs.acs.org/doi/10.1021/acs.analchem.9b01262)." *Analytical chemistry* 91.15 (2019): 9724-9731.
        - Ching Tarn, Wen-Feng Zeng. "[pDeep3: Toward More Accurate Spectrum Prediction with Fast Few-Shot Learning](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.0c05427)." *Analytical chemistry* 2021.
![](https://pubs.acs.org/cms/10.1021/acs.analchem.0c05427/asset/images/medium/ac0c05427_0006.gif)

2. **Prosit**
  - Code: [https://github.com/kusterlab/prosit](https://github.com/kusterlab/prosit)
  - [Webserver](https://www.proteomicsdb.org/prosit)
  - Reference:
        - Gessulat, Siegfried, et al. "[Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning](https://www.nature.com/articles/s41592-019-0426-7)." *Nature methods* 16.6 (2019): 509-518.
  - Application:
        - Verbruggen, Steven, et al. "[Spectral prediction features as a solution for the search space size problem in proteogenomics](https://doi.org/10.1016/j.mcpro.2021.100076)." Molecular & Cellular Proteomics (2021): 100076.
        - Wilhelm, M., Zolg, D.P., Graber, M. et al. [Deep learning boosts sensitivity of mass spectrometry-based immunopeptidomics](https://www.nature.com/articles/s41467-021-23713-9). Nat Commun 12, 3346 (2021).

3. **DeepMass**
  - Code: [https://github.com/verilylifesciences/deepmass](https://github.com/verilylifesciences/deepmass)
        - Prism is provided as a service using Google Cloud Machine Learning Engine.
  - Reference:
        - Tiwary, Shivani, et al. "[High-quality MS/MS spectrum prediction for data-dependent and data-independent acquisition data analysis](https://www.nature.com/articles/s41592-019-0427-6/)." *Nature methods* 16.6 (2019): 519-525.

4. **Predfull**
  - Code: [https://github.com/lkytal/PredFull](https://github.com/lkytal/PredFull)
  - Reference:
        - Liu, Kaiyuan, et al. "[Full-Spectrum Prediction of Peptides Tandem Mass Spectra using Deep Neural Network](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04867)." *Analytical Chemistry* 92.6 (2020): 4275-4283.

5. **Guan et al.**
  - Code: [https://zenodo.org/record/2652602#.X16LZZNKhT](https://zenodo.org/record/2652602#.X16LZZNKhT)
  - Reference:
        - Guan, Shenheng, Michael F. Moran, and Bin Ma. "[Prediction of LC-MS/MS properties of peptides from sequence by deep learning](https://www.mcponline.org/content/18/10/2099.full)." *Molecular & Cellular Proteomics* 18.10 (2019): 2099-2107.

6. **MS<sup>2</sup>CNN**
  - Code: [https://github.com/changlabtw/MS2CNN](https://github.com/changlabtw/MS2CNN)
  - Reference:
        - Lin, Yang-Ming, Ching-Tai Chen, and Jia-Ming Chang. "[MS2CNN: predicting MS/MS spectrum based on protein sequence using deep convolutional neural networks](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6297-6)." *BMC genomics* 20.9 (2019): 1-10.

7. **DeepDIA**
  - Code: [https://github.com/lmsac/DeepDIA/](https://github.com/lmsac/DeepDIA/)
  - Reference:
        - Yang, Yi, et al. "[In silico spectral libraries by deep learning facilitate data-independent acquisition proteomics](https://www.nature.com/articles/s41467-019-13866-z)." *Nature communications* 11.1 (2020): 1-11.

8. **pDeepXL**:
  - Code: [https://github.com/pFindStudio/pDeepXL](https://github.com/pFindStudio/pDeepXL)
  - Reference:
		- Chen, Zhen-Lin, et al. "[pDeepXL: MS/MS Spectrum Prediction for Cross-Linked Peptide Pairs by Deep Learning](https://pubs.acs.org/doi/10.1021/acs.jproteome.0c01004)." *J. Proteome Res.* 2021.

9. **Alpha-Frag**:
  - Code: [https://github.com/YuAirLab/Alpha-Frag](https://github.com/YuAirLab/Alpha-Frag)
  - Reference:
        - Jian, Song, et al. "[Alpha-Frag: a deep neural network for fragment presence prediction improves peptide identification by data independent acquisition mass spectrometry](https://doi.org/10.1101/2021.04.07.438629)." *bioRxiv.* 2021.
		
10. **Prosit Transformer**:
  - Code: N/A
  - Reference:
        - Jian, Song, et al. "[Prosit Transformer: A transformer for Prediction of MS2 Spectrum Intensities](https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00870)." *Journal of Proteome Research* 2022.

11. **PrAI-frag**
  - Code: [https://github.com/bertis-prai/PrAI-frag](https://github.com/bertis-prai/PrAI-frag)
  - [Webserver](http://www.prai.co.kr/)
  - Reference:
        - HyeonSeok Shin, Youngmin Park, Kyunggeun Ahn, and Sungsoo Kim "[Accurate Prediction of y Ions in Beam-Type Collision-Induced Dissociation Using Deep Learning](https://pubs.acs.org/doi/10.1021/acs.analchem.1c03184)." *Analytical Chemistry* May 24, 2022.
	
---
### Peptide retention time prediction

1. **AutoRT**
  - Code: [https://github.com/bzhanglab/AutoRT](https://github.com/bzhanglab/AutoRT)
  - Reference:
         - Wen, Bo, et al. "[Cancer neoantigen prioritization through sensitive and reliable proteogenomics analysis](https://www.nature.com/articles/s41467-020-15456-w)." *Nature communications* 11.1 (2020): 1-14.
  - Application:
         - Li, Kai, et al. "[DeepRescore: Leveraging Deep Learning to Improve Peptide Identification in Immunopeptidomics](https://doi.org/10.1002/pmic.201900334)." Proteomics 20.21-22 (2020): 1900334.
         - Rivero-Hinojosa, S., Grant, M., Panigrahi, A. et al. [Proteogenomic discovery of neoantigens facilitates personalized multi-antigen targeted T cell immunotherapy for brain tumors](https://doi.org/10.1038/s41467-021-26936-y). *Nat Commun* 12, 6689 (2021).
         - Daisha Van Der Watt, Hannah Boekweg, Thy Truong, Amanda J Guise, Edward D Plowey, Ryan T Kelly, Samuel H Payne. "[Benchmarking PSM identification tools for single cell proteomics](https://doi.org/10.1101/2021.08.17.456676)." *bioRxiv* 2021.
         - Jiang W, Wen B, Li K, et al. "[Deep learning-derived evaluation metrics enable effective benchmarking of computational tools for phosphopeptide identification](https://doi.org/10.1016/j.mcpro.2021.100171)." *Molecular & Cellular Proteomics*, 2021: 100171.
	 - Nekrakalaya, Bhagya, et al. "[Towards Phytopathogen Diagnostics? Coconut Bud Rot Pathogen Phytophthora palmivora Mycelial Proteome Analysis Informs Genome Annotation](https://pubmed.ncbi.nlm.nih.gov/35353641/)." *OMICS: A Journal of Integrative Biology* (2022).
        - Eric B Zheng, Li Zhao. "[Systematic identification of unannotated ORFs in Drosophila reveals evolutionary heterogeneity](https://doi.org/10.1101/2022.04.04.486978)." *bioRxiv* 2022.
	- Xiang H, Zhang L, Bu F, Guan X, Chen L, Zhang H, Zhao Y, Chen H, Zhang W, Li Y, Lee LJ, Mei Z, Rao Y, Gu Y, Hou Y, Mu F, Dong X. [A Novel Proteogenomic Integration Strategy Expands the Breadth of Neo-Epitope Sources](https://doi.org/10.3390/cancers14123016). **Cancers**. 2022; 14(12):3016.
		

2. **Prosit**
  - Code: [https://github.com/kusterlab/prosit](https://github.com/kusterlab/prosit)
  - [Webserver](https://www.proteomicsdb.org/prosit)
  - Reference:
        - Gessulat, Siegfried, et al. "[Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning](https://www.nature.com/articles/s41592-019-0426-7)." *Nature methods* 16.6 (2019): 509-518.
  - Application:
        - Wilhelm, M., Zolg, D.P., Graber, M. et al. [Deep learning boosts sensitivity of mass spectrometry-based immunopeptidomics](https://www.nature.com/articles/s41467-021-23713-9). Nat Commun 12, 3346 (2021).

3. **DeepMass**
  - Host: [https://github.com/verilylifesciences/deepmass](https://github.com/verilylifesciences/deepmass)
  - DeepMass::Prism is provided as a service using Google Cloud Machine Learning Engine.
  - Reference:
        - Tiwary, Shivani, et al. "[High-quality MS/MS spectrum prediction for data-dependent and data-independent acquisition data analysis](https://www.nature.com/articles/s41592-019-0427-6/)." *Nature methods* 16.6 (2019): 519-525.

4. **Guan et al.**
  - Code: [https://zenodo.org/record/2652602#.X16LZZNKhT](https://zenodo.org/record/2652602#.X16LZZNKhT)
  - Reference:
        - Guan, Shenheng, Michael F. Moran, and Bin Ma. "[Prediction of LC-MS/MS properties of peptides from sequence by deep learning](https://www.mcponline.org/content/18/10/2099.full)." *Molecular & Cellular Proteomics* 18.10 (2019): 2099-2107.

5. **DeepDIA**:
  - Code: [https://github.com/lmsac/DeepDIA](https://github.com/lmsac/DeepDIA)
  - Reference:
        - Yang, Yi, et al. "[In silico spectral libraries by deep learning facilitate data-independent acquisition proteomics](https://www.nature.com/articles/s41467-019-13866-z)." *Nature communications* 11.1 (2020): 1-11.

6. **DeepRT**:
  - Code: [https://github.com/horsepurve/DeepRTplus](https://github.com/horsepurve/DeepRTplus)
  - Reference:
        - Ma, Chunwei, et al. "[Improved peptide retention time prediction in liquid chromatography through deep learning](https://pubs.acs.org/doi/10.1021/acs.analchem.8b02386)." *Analytical chemistry* 90.18 (2018): 10881-10888.

7. **DeepLC**:
  - Code: [https://github.com/compomics/DeepLC](https://github.com/compomics/DeepLC)
  - Reference:
        - Bouwmeester, R., Gabriels, R., Hulstaert, N. et al. [DeepLC can predict retention times for peptides that carry as-yet unseen modifications](https://doi.org/10.1038/s41592-021-01301-5). *Nat Methods* 18, 1363–1369 (2021). 

8. **xiRT**:
  - Code: [https://github.com/Rappsilber-Laboratory/xiRT](https://github.com/Rappsilber-Laboratory/xiRT)
  - Reference:
        - Giese, S.H., Sinn, L.R., Wegner, F. et al. [Retention time prediction using neural networks increases identifications in crosslinking mass spectrometry](https://www.nature.com/articles/s41467-021-23441-0). *Nat Commun* 12, 3237 (2021).

---
### Peptide CCS prediction

1. **DeepCollisionalCrossSection**:
  - Code: [https://github.com/theislab/DeepCollisionalCrossSection](https://github.com/theislab/DeepCollisionalCrossSection)
  - Reference:
        - Meier, F., Köhler, N.D., Brunner, AD. et al. [Deep learning the collisional cross sections of the peptide universe from a million experimental values](https://www.nature.com/articles/s41467-021-21352-8). Nat Commun 12, 1185 (2021).

---
### Peptide detectability prediction

1. **CapsNet_CBAM**:
  - Code: [yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module](https://github.com/yuminzhe/yuminzhe-Prediction-of-peptide-detectability-based-on-CapsNet-and-CBAM-module)
  - Reference:
        - Yu M, Duan Y, Li Z, Zhang Y. [Prediction of Peptide Detectability Based on CapsNet and Convolutional Block Attention Module](https://doi.org/10.3390/ijms222112080). *International Journal of Molecular Sciences*. 2021; 22(21):12080. 

---
### MS/MS spectrum quality prediction

1. **SPEQ**:
  - Code: [https://github.com/sor8sh/SPEQ](https://github.com/sor8sh/SPEQ)
  - Reference:
        - Soroosh Gholamizoj, Bin Ma. [SPEQ: Quality Assessment of Peptide Tandem Mass Spectra with Deep Learning](https://doi.org/10.1093/bioinformatics/btab874). *Bioinformatics*. 2022; btab874.

---
### Peptide identification

1. **DeepNovo**: De novo peptide sequencing
  - Code: [https://github.com/nh2tran/DeepNovo](https://github.com/nh2tran/DeepNovo)
  - Reference:
        - Tran, Ngoc Hieu, et al. "[De novo peptide sequencing by deep learning](https://www.pnas.org/content/114/31/8247)." *Proceedings of the National Academy of Sciences* 114.31 (2017): 8247-8252.
![](https://www.pnas.org/cms/10.1073/pnas.1705691114/asset/5ad20657-c0b5-4904-b65c-14a880d768e7/assets/graphic/pnas.1705691114fig01.jpeg)

2. **DeepNovo-DIA**: De novo peptide sequencing
  - Code: [https://github.com/nh2tran/DeepNovo-DIA](https://github.com/nh2tran/DeepNovo-DIA)
  - Reference:
        - Tran, Ngoc Hieu, et al. "[Deep learning enables de novo peptide sequencing from data-independent-acquisition mass spectrometry](https://www.nature.com/articles/s41592-018-0260-3)." *Nature methods* 16.1 (2019): 63-66.

3. **SMSNet**: De novo peptide sequencing
  - Code: [https://github.com/cmb-chula/SMSNet](https://github.com/cmb-chula/SMSNet)
  - Reference:
        - Karunratanakul, Korrawe, et al. "[Uncovering thousands of new peptides with sequence-mask-search hybrid de novo peptide sequencing framework](https://www.mcponline.org/content/18/12/2478)." *Molecular & Cellular Proteomics* 18.12 (2019): 2478-2491.

4. **DeepRescore**: Leveraging deep learning to improve peptide identification
  - Code: [https://github.com/bzhanglab/DeepRescore](https://github.com/bzhanglab/DeepRescore)
  - Reference:
        - Li, Kai, et al. "[DeepRescore: Leveraging Deep Learning to Improve Peptide Identification in Immunopeptidomics](https://doi.org/10.1002/pmic.201900334)." Proteomics 20.21-22 (2020): 1900334.

5. **PointNovo**: De novo peptide sequencing
  - Code: [https://github.com/volpato30/PointNovo](https://github.com/volpato30/PointNovo)
  - Reference:
        - Qiao, R., Tran, N.H., Xin, L. et al. "[Computationally instrument-resolution-independent de novo peptide sequencing for high-resolution devices](https://doi.org/10.1038/s42256-021-00304-3)." *Nat Mach Intell* 3, 420–425 (2021). 

6. **pValid 2**: Leveraging deep learning to improve peptide identification
  - Reference:
        - Zhou, Wen-Jing, et al. "[pValid 2: A deep learning based validation method for peptide identification in shotgun proteomics with increased discriminating power](https://www.sciencedirect.com/science/article/abs/pii/S1874391921003134)." *Journal of Proteomics* (2021): 104414.
![](https://ars.els-cdn.com/content/image/1-s2.0-S1874391921003134-gr1.jpg)

7. **Casanovo**: De novo peptide sequencing
  - Code: [https://github.com/Noble-Lab/casanovo](https://github.com/Noble-Lab/casanovo)
  - Reference:
        - Melih Yilmaz, William E. Fondrie, Wout Bittremieux, Sewoong Oh, William Stafford Noble. "[De novo mass spectrometry peptide sequencing with a transformer model](https://doi.org/10.1101/2022.02.07.479481)". *bioRxiv*. 2022.
![](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

8. **PepNet**: De novo peptide sequencing
  - Code: [https://github.com/lkytal/PepNet](https://github.com/lkytal/PepNet)
  - Reference:
        - Kaiyuan Liu, Yuzhen Ye, Haixu Tang. "[PepNet: A Fully Convolutional Neural Network for De novo Peptide Sequencing](https://www.researchsquare.com/article/rs-1341615/v1)". *Research Square*. 2022.
![](https://github.com/lkytal/PepNet/raw/master/imgs/model.png)

9. **DePS**: De novo peptide sequencing
  - Code: N/A
  - Reference:
        - Cheng Ge, Yi Lu, Jia Qu, Liangxu Xie, Feng Wang, Hong Zhang, Ren Kong, Shan Chang. "[DePS: An improved deep learning model for de novo peptide sequencing](https://arxiv.org/pdf/2203.08820.pdf)". *arXiv*. 2022.
![](https://www.researchgate.net/publication/359310015/figure/fig1/AS:1134819052388352@1647573354122/The-process-of-feature-extraction-Where-prefix-max-is-an-iterative-variable-with-an.png)
![](https://www.researchgate.net/publication/359310015/figure/fig2/AS:1134819052400640@1647573354139/DePS-net-It-consists-mainly-of-three-one-dimensional-convolutional-layers-a-deep.png)

10. **DeepSCP**: Utilizing deep learning to boost single-cell proteome coverage
  - Code: [https://github.com/XuejiangGuo/DeepSCP](https://github.com/XuejiangGuo/DeepSCP)
  - Reference:
        - Bing Wang, Yue Wang, Yu Chen, Mengmeng Gao, Jie Ren, Yueshuai Guo, Chenghao Situ, Yaling Qi, Hui Zhu, Yan Li, Xuejiang Guo, [DeepSCP: utilizing deep learning to boost single-cell proteome coverage](https://doi.org/10.1093/bib/bbac214). Briefings in Bioinformatics, 2022;, bbac214.
![](https://github.com/XuejiangGuo/DeepSCP/raw/main/Figure1.jpg)

---
### Data-independent acquisition mass spectrometry

1. **Alpha-XIC**
  - Code: [https://github.com/YuAirLab/Alpha-XIC](https://github.com/YuAirLab/Alpha-XIC)
  - Reference:
        - Jian Song, Changbin Yu. "[Alpha-XIC: a deep neural network for scoring the coelution of peak groups improves peptide identification by data-independent acquisition mass spectrometry](https://doi.org/10.1093/bioinformatics/btab544)." *Bioinformatics*, btab544 (2021). 

2. **DeepDIA**:
  - Code: [https://github.com/lmsac/DeepDIA](https://github.com/lmsac/DeepDIA)
  - Reference:
        - Yang, Yi, et al. "[In silico spectral libraries by deep learning facilitate data-independent acquisition proteomics](https://www.nature.com/articles/s41467-019-13866-z)." *Nature communications* 11.1 (2020): 1-11.

3. **DeepPhospho**: impoves spectral library generation for DIA phosphoproteomics
  - Code: [https://github.com/weizhenFrank/DeepPhospho](https://github.com/weizhenFrank/DeepPhospho)
  - Reference: 
        - Lou, R., Liu, W., Li, R. et al. [DeepPhospho accelerates DIA phosphoproteome profiling through in silico library generation](https://doi.org/10.1038/s41467-021-26979-1). *Nat Commun* 12, 6685 (2021). 
		
---
### Protein post-translational modification site prediction

1. **DeepACE**:a tool for predicting lysine acetylation sites which belong of PTM questions. 
  - Code: [https://github.com/jiagenlee/DeepAce](https://github.com/jiagenlee/DeepAce)
  - Reference:
        - Zhao, Xiaowei, et al. "[General and species-specific lysine acetylation site prediction using a bi-modal deep architecture](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8487006)." *IEEE Access* 6 (2018): 63560-63569.

2. **[Deep-PLA](http://deeppla.cancerbio.info/)**: for prediction of HAT/HDAC-specific acetylation
  - [Webserver](http://deeppla.cancerbio.info/webserver.php)
  - Reference:
        - "[Deep learning based prediction of reversible HAT/HDAC-specific lysine acetylation](https://doi.org/10.1093/bib/bbz107)." *Briefings in Bioinformatics* (2019).

3. **DeepAcet**: to predict the lysine acetylation sites in protein
  - Code: [https://github.com/Sunmile/DeepAcet](https://github.com/Sunmile/DeepAcet)
  - Reference:
        - "[A deep learning method to more accurately recall known lysine acetylation sites](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2632-9)." *BMC bioinformatics* 20.1 (2019): 49.

4. **DNNAce**
  - Code: [https://github.com/QUST-AIBBDRC/DNNAce](https://github.com/QUST-AIBBDRC/DNNAce)
  - Reference:
        - "[DNNAce: Prediction of prokaryote lysine acetylation sites through deep neural networks with multi-information fusion](https://www.sciencedirect.com/science/article/pii/S0169743919305453)." *Chemometrics and Intelligent Laboratory Systems* (2020): 103999.

5. **DeepKcr**
  - Code: [https://github.com/linDing-group/Deep-Kcr](https://github.com/linDing-group/Deep-Kcr)  
  - Reference:
        - "[Deep-Kcr: Accurate detection of lysine crotonylation sites using deep learning method](https://academic.oup.com/bib/article/22/4/bbaa255/5937175)", Briefings in Bioinformatics, Volume 22, Issue 4, July 2021.
        - "[Identification of Protein Lysine Crotonylation Sites by a Deep Learning Framework With Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8959202)." *IEEE Access* 8 (2020): 14244-14252.

6. **DeepGly**
  - Reference:  
        - Chen, Jingui, et al. "[DeepGly: A Deep Learning Framework With Recurrent and Convolutional Neural Networks to Identify Protein Glycation Sites From Imbalanced Data](https://ieeexplore.ieee.org/abstract/document/8852736)." *IEEE Access* 7 (2019): 142368-142378.

7. **Longetal2018**
  - Reference:  
        - Long, Haixia, et al. "[A hybrid deep learning model for predicting protein hydroxylation sites](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6164125/)." *International Journal of Molecular Sciences* 19.9 (2018): 2817.

8. **[MUscADEL](http://muscadel.erc.monash.edu/)**
  - Reference:  
        - Chen, Zhen, et al. "[Large-scale comparative assessment of computational predictors for lysine post-translational modification sites](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6954452/)." *Briefings in bioinformatics* 20.6 (2019): 2267-2290.

9. **LEMP**
  - Reference:
        - Chen, Zhen, et al. "[Integration of a deep learning classifier with a random forest approach for predicting malonylation sites](ncbi.nlm.nih.gov/pmc/articles/PMC6411950/)." *Genomics, proteomics & bioinformatics* 16.6 (2018): 451-459.

10. **[DeepNitro](http://deepnitro.renlab.org/)**
  - Reference:
        - Xie, Yubin, et al. "[DeepNitro: prediction of protein nitration and nitrosylation sites by deep learning](https://www.sciencedirect.com/science/article/pii/S1672022918303474)." *Genomics, proteomics & bioinformatics* 16.4 (2018): 294-306.

11. **MusiteDeep**
  - Code: [https://github.com/duolinwang/MusiteDeep](https://github.com/duolinwang/MusiteDeep)
  - Reference:
        - Wang, Duolin, et al. "[MusiteDeep: a deep-learning framework for general and kinase-specific phosphorylation site prediction](https://pubmed.ncbi.nlm.nih.gov/29036382/)." *Bioinformatics* 33.24 (2017): 3909-3916.

12. **[NetPhosPan](https://services.healthtech.dtu.dk/service.php?NetPhospan-1.0)**:Prediction of phosphorylation using CNNs
  - Reference:
        - Fenoy, Emilio, et al. "[A generic deep convolutional neural network framework for prediction of receptor–ligand interactions—NetPhosPan: application to kinase phosphorylation prediction](https://academic.oup.com/bioinformatics/article/35/7/1098/5088322)." *Bioinformatics* 35.7 (2019): 1098-1107.

13. **DeepPhos**
  - Code: [https://github.com/USTC-HIlab/DeepPhos](https://github.com/USTC-HIlab/DeepPhos)
  - Reference:
        - Luo, Fenglin, et al. "[DeepPhos: prediction of protein phosphorylation sites with deep learning](https://academic.oup.com/bioinformatics/article/35/16/2766/5270665)." *Bioinformatics* 35.16 (2019): 2766-2773.

14. **EMBER**
  - Code: [https://github.com/gomezlab/EMBER](https://github.com/gomezlab/EMBER)
  - Reference:
        - Kirchoff, Kathryn E., and Shawn M. Gomez. "[EMBER: Multi-label prediction of kinase-substrate phosphorylation events through deep learning](https://www.biorxiv.org/content/10.1101/2020.02.04.934216v1)." *BioRxiv* (2020).

15. **DeepKinZero**
  - Code: [https://github.com/tastanlab/DeepKinZero](https://github.com/tastanlab/DeepKinZero)
  - Reference:
        - Deznabi, Iman, et al. "[DeepKinZero: zero-shot learning for predicting kinase–phosphosite associations involving understudied kinases](https://academic.oup.com/bioinformatics/article/36/12/3652/5733725)." *Bioinformatics* 36.12 (2020): 3652-3661.

16. **CapsNet_PTM**: CapsNet for Protein Post-translational Modification site prediction.
  - Code: [https://github.com/duolinwang/CapsNet_PTM](https://github.com/duolinwang/CapsNet_PTM)
  - Reference:
        - Wang, Duolin, Yanchun Liang, and Dong Xu. "[Capsule network for protein post-translational modification site prediction](https://academic.oup.com/bioinformatics/article/35/14/2386/5232223)." *Bioinformatics* 35.14 (2019): 2386-2394.

17. **[GPS-Palm](http://gpspalm.biocuckoo.cn)**
  - Reference:
        - Ning, Wanshan, et al. "[GPS-Palm: a deep learning-based graphic presentation system for the prediction of S-palmitoylation sites in proteins](https://doi.org/10.1093/bib/bbaa038)." *Briefings in Bioinformatics* (2020).

18. **CNN-SuccSite**
  - Reference:
        - Huang, Kai-Yao, Justin Bo-Kai Hsu, and Tzong-Yi Lee. "[Characterization and Identification of Lysine Succinylation Sites based on Deep Learning Method](https://www.nature.com/articles/s41598-019-52552-4)." *Scientific reports* 9.1 (2019): 1-15.
![](https://www.researchgate.net/publication/337094467/figure/fig4/AS:959554468712469@1605787021560/Flow-chart-of-the-proposed-method-Four-major-steps-were-involved-such-as-construction-of.png)

19. **DeepUbiquitylation**
  - Code: [https://github.com/jiagenlee/deepUbiquitylation](https://github.com/jiagenlee/deepUbiquitylation)
  - Reference:
        - He, Fei, et al. "[Large-scale prediction of protein ubiquitination sites using a multimodal deep architecture](https://link.springer.com/article/10.1186/s12918-018-0628-0)." *BMC systems biology* 12.6 (2018): 109.

20. **DeepUbi**
  - Code: [https://github.com/Sunmile/DeepUbi](https://github.com/Sunmile/DeepUbi)
  - Reference: 
        - Fu, Hongli, et al. "[DeepUbi: a deep learning framework for prediction of ubiquitination sites in proteins](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2677-9)." *BMC bioinformatics* 20.1 (2019): 1-10.

21. **[Sohoko-Kcr](https://sohoko-research-9uu23.ondigitalocean.app/)**
  - [Webserver](https://sohoko-research-9uu23.ondigitalocean.app/#/webserver)
  - Reference:
        - Sian Soo Tng, et al. "[Improved Prediction Model of Protein Lysine Crotonylation Sites Using Bidirectional Recurrent Neural Networks
](https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00848)." *J. Proteome Res.* 2021.

---
### MHC-peptide binding prediction

1. **ConvMHC**
  - Reference:
        - Han, Youngmahn, and Dongsup Kim. "[Deep convolutional neural networks for pan-specific peptide-MHC class I binding prediction](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1997-x)." *BMC bioinformatics* 18.1 (2017): 585.

2. **HLA-CNN**
  - Code: [https://github.com/uci-cbcl/HLA-bind](https://github.com/uci-cbcl/HLA-bind)
  - Reference:
        - Vang, Yeeleng S., and Xiaohui Xie. "[HLA class I binding prediction via convolutional neural networks](https://academic.oup.com/bioinformatics/article/33/17/2658/3746909)." *Bioinformatics* 33.17 (2017): 2658-2665.

3. **[DeepMHC](http://mleg.cse.sc.edu/deepMHC)**
  - [Web services](http://mleg.cse.sc.edu/software)
  - Reference:
        - Hu, Jianjun, and Zhonghao Liu. "[DeepMHC: Deep convolutional neural networks for high-performance peptide-MHC binding affinity prediction](https://doi.org/10.1101/239236)." *bioRxiv* (2017): 239236.

4. **DeepSeqPan**: Prediction of peptide-MHC bindings
  - Code: [https://github.com/pcpLiu/DeepSeqPan](https://github.com/pcpLiu/DeepSeqPan)
  - Reference:  
        - Liu, Zhonghao, et al. "[DeepSeqPan, a novel deep convolutional neural network model for pan-specific class I HLA-peptide binding affinity prediction](https://www.nature.com/articles/s41598-018-37214-1)." *Scientific reports* 9.1 (2019): 1-10.

5. **AI-MHC**
  - [Webserver](http://baras.pathology.jhu.edu/AI-MHC/index.html)
  - Reference:  
        - Sidhom, John-William, Drew Pardoll, and Alexander Baras. "[AI-MHC: an allele-integrated deep learning framework for improving Class I & Class II HLA-binding predictions](https://www.biorxiv.org/content/10.1101/318881v1.full)." *bioRxiv* (2018): 318881.

6. **DeepSeqPanII**
  - Code: [https://github.com/pcpLiu/DeepSeqPanII](https://github.com/pcpLiu/DeepSeqPanII)
  - Reference:  
        - Liu, Zhonghao, et al. "[DeepSeqPanII: an interpretable recurrent neural network model with attention mechanism for peptide-HLA class II binding prediction](https://www.biorxiv.org/content/10.1101/817502v1)." *bioRxiv* (2019): 817502.
![](https://github.com/pcpLiu/DeepSeqPanII/raw/master/model.png)

7. **MHCSeqNet**
  - Code: [https://github.com/cmb-chula/MHCSeqNet](https://github.com/cmb-chula/MHCSeqNet)
  - Reference:  
        - Phloyphisut, Poomarin, et al. "[MHCSeqNet: a deep neural network model for universal MHC binding prediction](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2892-4)." *BMC bioinformatics* 20.1 (2019): 270.

8. **[MARIA](https://maria.stanford.edu/)**
  - Reference:
        - Chen, Binbin, et al. "[Predicting HLA class II antigen presentation through integrated deep learning](https://www.nature.com/articles/s41587-019-0280-2)." *Nature biotechnology* 37.11 (2019): 1332-1343.

9. **MHCflurry**
  - Code: [https://github.com/openvax/mhcflurry](https://github.com/openvax/mhcflurry)
  - Reference:
        - T. O'Donnell, A. Rubinsteyn, U. Laserson. "[MHCflurry 2.0: Improved pan-allele prediction of MHC I-presented peptides by incorporating antigen processing](https://doi.org/10.1016/j.cels.2020.06.010)," Cell Systems, 2020.
        - O'Donnell, Timothy J., et al. "[MHCflurry: open-source class I MHC binding affinity prediction](https://www.sciencedirect.com/science/article/pii/S2405471218302321)." *Cell systems* 7.1 (2018): 129-132.

10. **[DeepHLApan](http://biopharm.zju.edu.cn/deephlapan/)**
  - Code: [https://github.com/jiujiezz/deephlapan](https://github.com/jiujiezz/deephlapan)
  - Reference:
        - Wu, Jingcheng, et al. "[DeepHLApan: a deep learning approach for neoantigen prediction considering both HLA-peptide binding and immunogenicity](https://www.frontiersin.org/articles/10.3389/fimmu.2019.02559/full)." *Frontiers in Immunology* 10 (2019): 2559.
![](http://biopharm.zju.edu.cn/deephlapan/wp-content/uploads/2019/06/Figure-2-1024x453.png)

11. **ACME**
  - Code: [https://github.com/HYsxe/ACME](https://github.com/HYsxe/ACME)
  - Reference:
        - Hu, Yan, et al. "[ACME: pan-specific peptide–MHC class I binding prediction through attention-based deep neural networks](https://www.biorxiv.org/content/10.1101/468363v1.full)." *Bioinformatics* 35.23 (2019): 4946-4954.
![](https://www.biorxiv.org/content/biorxiv/early/2018/11/14/468363/F1.large.jpg?width=800&height=600&carousel=1)

12. **EDGE**
  - Code: [Supplementary data](https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.4313/MediaObjects/41587_2019_BFnbt4313_MOESM48_ESM.zip)
  - Reference:
        - Bulik-Sullivan, Brendan, et al. "[Deep learning using tumor HLA peptide mass spectrometry datasets improves neoantigen identification](https://www.nature.com/articles/nbt.4313)." *Nature biotechnology* 37.1 (2019): 55-63.

13. **MHC-I**
  - Code: [https://github.com/zty2009/MHC-I](https://github.com/zty2009/MHC-I)
  - Reference:
        - Zhao, Tianyi, et al. "[Peptide-Major Histocompatibility Complex Class I Binding Prediction Based on Deep Learning With Novel Feature](https://www.frontiersin.org/articles/10.3389/fgene.2019.01191/full)." *Frontiers in Genetics* 10 (2019).

14. **[MHCnuggets](https://karchinlab.org/apps/appMHCnuggets.html)**
  - Code: [https://github.com/KarchinLab/mhcnuggets](https://github.com/KarchinLab/mhcnuggets)
  - Reference:
        - Shao, Xiaoshan M., et al. "[High-throughput prediction of MHC class i and ii neoantigens with MHCnuggets](https://cancerimmunolres.aacrjournals.org/content/8/3/396.abstract)." *Cancer Immunology Research* 8.3 (2020): 396-408.

15. **[DeepNeo](https://omics.kaist.ac.kr/resources)**
  - Code: [DeepNeo-MHC](https://github.com/sysbioinfo/DeepNeo-MHC)
  - Reference:
        - Kim, Kwoneel, et al. "[Predicting clinical benefit of immunotherapy by antigenic or functional mutations affecting tumour immunogenicity](https://www.nature.com/articles/s41467-020-14562-z)." *Nature communications* 11.1 (2020): 1-11.

16. **DeepLigand**
  - Code: [https://github.com/gifford-lab/DeepLigand](https://github.com/gifford-lab/DeepLigand)
  - Reference:
        - Zeng, Haoyang, and David K. Gifford. "[DeepLigand: accurate prediction of MHC class I ligands using peptide embedding](https://academic.oup.com/bioinformatics/article/35/14/i278/5529131)." *Bioinformatics* 35.14 (2019): i278-i283.

17. **PUFFIN**
  - Code: [http://github.com/gifford-lab/PUFFIN](http://github.com/gifford-lab/PUFFIN)
  - Reference:
        - Zeng, Haoyang, and David K. Gifford. "[Quantification of uncertainty in peptide-MHC binding prediction improves high-affinity peptide Selection for therapeutic design](https://doi.org/10.1016/j.cels.2019.05.004)." *Cell systems* 9.2 (2019): 159-166.

18. **NeonMHC2**
  - [Webserver](https://neonmhc2.org/)
  - Code: [https://bitbucket.org/dharjanto-neon/neonmhc2](https://bitbucket.org/dharjanto-neon/neonmhc2)
  - Reference:
        - Abelin, Jennifer G., et al. "[Defining HLA-II ligand processing and binding rules with mass spectrometry enhances cancer epitope prediction](https://www.sciencedirect.com/science/article/pii/S1074761319303632)." *Immunity* 51.4 (2019): 766-779.

19. **USMPep**
  - Code: [https://github.com/nstrodt/USMPep](https://github.com/nstrodt/USMPep)
  - Reference:
        - Vielhaben, Johanna, et al. "[USMPep: universal sequence models for major histocompatibility complex binding affinity prediction](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03631-1#Sec14)." *BMC bioinformatics* 21.1 (2020): 1-16.

20. **MHCherryPan**
  - Reference:
        - Xie, Xuezhi, Yuanyuan Han, and Kaizhong Zhang. "[MHCherryPan. a novel model to predict the binding affinity of pan-specific class I HLA-peptide](https://ieeexplore.ieee.org/abstract/document/8982962)." 2019 *IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE*, 2019.

21. **DeepAttentionPan**
  - Code: [https://github.com/jjin49/DeepAttentionPan](https://github.com/jjin49/DeepAttentionPan)
  - Reference:
        - Jin, Jing, et al. "[Attention mechanism-based deep learning pan-specific model for interpretable MHC-I peptide binding prediction](https://doi.org/10.1101/830737)." *bioRxiv* (2019): 830737.

---
### Benchmarking

1. Xu R, Sheng J, Bai M, et al. "[A comprehensive evaluation of MS/MS spectrum prediction tools for shotgun proteomics](https://doi.org/10.1002/pmic.201900345)". *Proteomics*, 2020, 20(21-22): 1900345.
2. Wenrong Chen, Elijah N. McCool, Liangliang Sun, Yong Zang, Xia Ning, Xiaowen Liu. "[Evaluation of Machine Learning Models for Proteoform Retention and Migration Time Prediction in Top-Down Mass Spectrometry](https://pubs.acs.org/doi/10.1021/acs.jproteome.2c00124)". *J. Proteome Res.* (2022).
3. Emily Franklin, Hannes L. Röst, "[Comparing Machine Learning Architectures for the Prediction of Peptide Collisional Cross Section](https://doi.org/10.1101/2022.03.01.482566)". *bioRxiv* (2022).

---
### Reviews about deep learning in proteomics

1. Wen, B., Zeng, W.-F., Liao, Y., Shi, Z., Savage, S. R., Jiang, W., Zhang, B., "[Deep Learning in Proteomics](https://doi.org/10.1002/pmic.201900335)". *Proteomics* 2020, 20, 1900335.
2. Meyer, Jesse G. "[Deep learning neural network tools for proteomics](https://doi.org/10.1016/j.crmeth.2021.100003)". *Cell Reports Methods* (2021): 100003.
3. Matthias Mann, Chanchal Kumar, Wen-Feng Zeng, Maximilian T. Strauss, [Artificial intelligence for proteomics and biomarker discovery](https://doi.org/10.1016/j.cels.2021.06.006). *Cell Systems* 12, August 18, 2021.
4. Yang, Y., Lin L., Qiao L., "[Deep learning approaches for data-independent acquisition proteomics](https://doi.org/10.1080/14789450.2021.2020654)". *Expert Review of Proteomics* 17 Dec 2021.

---
## Virus Identification

### ViraMiner
**Paper:** [ViraMiner: Deep learning on raw DNA sequences for identifying viral genomes in human samples](https://pubmed.ncbi.nlm.nih.gov/31509583/)<br>
**Code:** [https://github.com/NeuroCSUT/ViraMiner](https://github.com/NeuroCSUT/ViraMiner)<br>
![](https://www.ncbi.nlm.nih.gov/pmc/articles/instance/6738585/bin/pone.0222271.g006.jpg)

---
### [SAR-CoV-2]( https://www.ncbi.nlm.nih.gov/sars-cov-2/)
**Database:** [SARS-CoV-2, taxid:2697049 (Nucleotide)](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=SARS-CoV-2,%20taxid:2697049) <br>
* SARS-CoV-2 related compounds, substances, pathways, bioassays, and more in [PubChem](https://pubchemdocs.ncbi.nlm.nih.gov/covid-19)
  - [Compounds](https://pubchem.ncbi.nlm.nih.gov/#tab=compound&query=covid-19%20clinicaltrials) used in SARS-CoV-2 clinical trials
  - [Compounds](https://pubchem.ncbi.nlm.nih.gov/#tab=compound&query=covid-19%20protein%20data%20bank) found in COVID19-related PDB structures

---
### SARS-CoV-2 accurate identification
**Paper:** [Accurate Identification of SARS-CoV-2 from Viral Genome Sequences using Deep Learning](https://www.biorxiv.org/content/10.1101/2020.03.13.990242v1.full.pdf)<br>
**Code:** [https://github.com/albertotonda/deep-learning-coronavirus-genome](https://github.com/albertotonda/deep-learning-coronavirus-genome)<br>
**Kaggle:** [rkuo2000/coronavirus-genome-identification](https://www.kaggle.com/code/rkuo2000/coronavirus-genome-identification)<br>

---
### SARS-CoV-2 primers
**Paper:** [Classification and specific primer design for accurate detection of SARS-CoV-2 using deep learning](https://www.nature.com/articles/s41598-020-80363-5)<br>
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-020-80363-5/MediaObjects/41598_2020_80363_Fig1_HTML.png?as=webp)
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-020-80363-5/MediaObjects/41598_2020_80363_Fig5_HTML.png?as=webp)
**Code:** [https://github.com/steppenwolf0/primers-sars-cov-2](https://github.com/steppenwolf0/primers-sars-cov-2)<br>

---
### [Coronavirus Typing Tool](ttps://www.genomedetective.com/app/typingtool/cov/)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Coronavirus_typing_tool.png?raw=true)

---
### [OpenVaccine COVID-19 mRNA Vaccine Degradation Prediction](https://www.kaggle.com/c/stanford-covid-vaccine)
![](https://storage.googleapis.com/kaggle-media/competitions/Stanford/banner%20(2).png)
**Kaggle:** [OpenVaccine: GCN (GraphSAGE)+GRU+KFold](https://www.kaggle.com/code/arjunashok33/openvaccine-gcn-graphsage-gru-kfold)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


