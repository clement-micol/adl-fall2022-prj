# ADL Final Project: 

Term: Fall 2022

+ Team #
+ Projec title: Detecting Cancer Metastases on Gigapixel Pathology Images
+ Team members
	+ Clement Micol (MC5104)
+ Project summary: The goal of this project is to detect Cancer Metastases (or tumor) on Gigapixel Pathology Images. To do so, we transform this problem from an object detection to a classification problem. Indeed, we split the gigapixel images into smaller patches of images where we looked whether this smaller patch has a Cancer Metastases pixel in it or not. Our model is then capable of generating a heatmap when fed a Pathology Images, detecting which part of the cells are the most likely to be cancerous!
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
