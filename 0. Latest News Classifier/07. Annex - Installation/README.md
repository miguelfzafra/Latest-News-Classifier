## Annex 1: Installation



In order to be able to replicate the results obtained in this project, we will detail the necessary installation and execution steps starting from a new virtual machine with **Ubuntu 18.04**.



Please note that the process of deploying the app to a web server is covered in the Annex 2: Web App Deployment with Heroku.



### Installation of the Ubuntu virtual machine



First of all, download and install the Ubuntu 18.04 disk image from [here](https://www.ubuntu.com/download/desktop).



### Installation of Anaconda distribution



The next step is to install the Anaconda distribution for Linux systems, which will have a lot of the libraries needed installed for us. It can be downloaded from [here](https://www.anaconda.com/download/#linux).



The installation instructions can be found [here](http://docs.anaconda.com/anaconda/install/linux/).



### Installation of additional python packages



Although a lot of packages will be installed with the Anaconda distribution, we will need to manually install other ones. They can be installed by typing the following commands in the Linux shell:



```

# altair
conda install -c conda-forge altair vega_datasets notebook vega

# dash
pip install dash==0.35.1
# dash HTML components
pip install dash-html-components==0.13.4

# dash core components
pip install dash-core-components==0.42.1

# dash table
pip install dash-table==3.1.11

```
	


### Installation of RStudio



We will also need RStudio for the Dataset creation step. Instructions on how to install R and RStudio on Ubuntu 18.04 can be found [here](https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-18-04) and [here](https://www.rstudio.com/products/rstudio/download/).



### Paths



Finally, please copy the github files in the root path of the system. For example, if our username is `lnc`, the main folder `0. Latest News Classifier` should be placed in `home/lnc/`. 


