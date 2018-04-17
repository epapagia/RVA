Reference Vector Algorithm

We provide the implementation of the Reference Vector Algorithm (RVA). 

Prerequisites: 

Go to the project of GloVe: Global Vectors for Word Representation on Github, https://github.com/stanfordnlp/GloVe 

Clone the project: 
$ git clone https://github.com/stanfordnlp/GloVe.git

and then:
$ cd glove && make

Replace the existing demo.sh file in the glove project with the demo.sh file of the current project and add RVA.py also in the glove project.

Run RVA.py providing in the code the folders with the corresponding fulltexts and abstracts, respectively.

e.g.
data_original_path = '/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/all_docs_abstacts_refined/'

abstracts_path = '/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin_Abstracts/'

Files 628247.txt, 628247.key, 628247.abstr are example data files that are given as input to the system. Particularly, the files 628247.txt, 628247.key contain the fulltext and its corresponding keywords, respectively and they should be put in the folder data_original_path whereas the file 628247.abstr contains only the title as well as the abstract and it should be located in the folder abstracts_path. 

