Reference Vector Algorithm

We provide the implementation of the Reference Vector Algorithm (RVA). 

Prerequisites: 

Go to the project of GloVe: Global Vectors for Word Representation on Github, https://github.com/stanfordnlp/GloVe 

Clone the project: 
$ git clone http://github.com/stanfordnlp/glove

and then:
$ cd glove && make

Replace the existing demo.sh file in the glove project with the demo.sh file of the current project.

Run RVA.py providing in the code the folders with the corresponding fulltexts and abstracts, respectively.

e.g.
data_original_path = '/home/eirini/Projects_Atypon/NLTKTutorial/SemEval2010-Maui/original/SemEval2010/'

abstracts_path = '/home/eirini/Projects_Atypon/NLTKTutorial/Semeval2010_Abstracts/'

