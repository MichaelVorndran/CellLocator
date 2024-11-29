## How to Download the App

Click on Releases

![grafik](https://github.com/user-attachments/assets/364e3cdd-d1fb-4082-9c75-7b297e528311)


Then download the zip file from the latest release and extract it. 

![grafik](https://github.com/user-attachments/assets/c5c1a90f-5f7b-410e-9b49-547a5b8a1e70)

Inside the folder you find a Quickstart Guide. 

![grafik](https://github.com/user-attachments/assets/78dd5a41-66fa-43c1-963e-647a2023e615)


# CellLocator


CellLocator is an open-source application designed to simplify and enhance live-cell image analysis for users of the Sartorius **Incucyte®** system. It provides automated cell segmentation, fluorescence quantification, and data export for cell viability studies and fluorescence kinetics.

## License

CellLocator is freely available under the MIT License. This means you can use, modify, and distribute this software and its source code for any purpose, even commercially, as long as you include the original copyright and license notice.

## Cite

If you use CellLocator in your research, please cite the software using the following DOI:

[![DOI](https://zenodo.org/badge/771184849.svg)](https://zenodo.org/doi/10.5281/zenodo.13774182)



![grafik](https://github.com/user-attachments/assets/2ad4ce6a-c2a7-457d-b053-5ea0c7a9780b)



![grafik](https://github.com/user-attachments/assets/43319672-e4d6-4aad-b418-f035608be15a)



![grafik](https://github.com/user-attachments/assets/9da690ff-a333-4111-a3f5-47fbc39d3f87)




## CellLocator in Research

CellLocator has already proven useful in scientific studies.

* **Pre-print:**
[Ferroptosis propagates to neighboring cells via cell-cell contacts](https://www.biorxiv.org/content/10.1101/2023.03.24.534081v1.abstract)

* **Published:**
[TBK1-associated adapters TANK and AZI2 protect mice against TNF-induced cell death and severe autoinflammatory diseases](https://www.nature.com/articles/s41467-024-54399-4)




## Project History

CellLocator originated in Spring 2021 when [Bernhard Röck](https://www.linkedin.com/in/bernhard-r%C3%B6ck/?originalSubdomain=de) and I ([Michael Vorndran](https://www.linkedin.com/in/michael-vorndran-541001271/)) began developing an AI-powered tool for analyzing brightfield microscopy images. Our initial vision was ambitious: a versatile platform for single-cell analysis that could also classify different types of cell death (ferroptosis, apoptosis, etc.). While we made some progress in this direction, creating a robust and reliable classifier for cell death types requires a vast amount of meticulously labeled training data. Unfortunately, due to resource limitations, we were unable to generate a sufficiently large and diverse dataset to achieve this goal with the desired accuracy. Consequently, while some preliminary cell death classification capabilities were explored in early prototypes, **this feature has been entirely removed from the open-source CellLocator application**. Our focus shifted to providing a highly accurate and efficient tool for cell segmentation, fluorescence quantification, and kinetic analysis, which we believe offers significant value to researchers even without cell death type classification. While our early work included some images from an ImageXpress® Micro 4 MD system, due to resource constraints and the widespread use of Incucyte® systems, we focused CellLocator's development and training specifically on Incucyte® brightfield images.

In 2022, our team (then called "Cell ImAIging") won a ["Start-up Your Idea"](https://www.cecad.uni-koeln.de/outreach/news/article/cecad-scientists-win-start-up-your-idea-competition/) competition and secured seed funding, followed by a [GO-Bio initial](https://www.go-bio.de/gobio/de/gefoerderte-projekte/gobio-initial/_documents/zelltodart.html) grant. Despite this promising start, we were unable to secure further funding in 2023 to continue the startup. Rather than abandoning the project, we decided to open-source CellLocator, making our powerful analysis tools freely available to the research community. CellLocator's deep learning models were trained using a novel method described in our paper on [“Inconsistency Masks”](https://arxiv.org/abs/2401.14387), enabling accurate segmentation and analysis even with limited training data. The robustness and effectiveness of CellLocator have already been demonstrated through its use in several scientific publications (link to biorxiv publication, link to nature publication).

We hope CellLocator empowers researchers to extract deeper insights from their live-cell imaging data, accelerating discoveries in cell biology and drug development.


### Early Prototype in Action

This video shows our very first prediction from April 6, 2021. We only had a tiny dataset of two or three images at this point, and the model was built on MobileNet. Humble beginnings, right?






https://github.com/user-attachments/assets/01c607df-32d2-4371-8f58-9dd5db0e7523


