## Deployment



Dash apps are web applications that use Flask as the web framework. One of the simplest ways to deploy an application is with **Heroku**.



The steps to deploy a web application with Heroku through `anaconda` in Windows are:



#### Sign in to Heroku



#### Open the Anaconda Prompt



#### Create a new folder (we have created it into **06. App Creation**)



```

$ mkdir dash-app-lnclass
$ cd dash-app-lnclass

```



#### Initialize the folder with git


```

$ git init 

```



#### Create an `environment.yml` file in `dash-app-lnclass`

```

name: dash_app_lnclass #Environment name
dependencies:
  - python=3.6
  - pip:
    - dash
    - dash-renderer
    - dash-core-components
    - dash-html-components
    - dash-table
    - plotly
    - gunicorn # for app deployment
    - nltk
    - scikit-learn
    - beautifulsoup4
    - requests
    - pandas
    - numpy
    - lxml

```



#### Create the environment from `environment.yml` and activate it

```
$ conda env create
$ activate dash_app_lnclass 

```


#### Initialize the folder with `app.py`, `requirements.txt` and a `Procfile`:

Procfile:

```
web: gunicorn app:server

```

Requirements:

```
# run this
$ pip freeze > requirements.txt

```

#### `nltk` and pickles

Since we will be using `nltk` downloads and pickles, we need to add the `nltk.txt` file  and the `Pickles` folder.

#### Initialize Heroku, add files to Git, and deploy

```
$ heroku create lnclass # change my-dash-app to a unique name
$ git add . # add all files to git
$ git commit -m 'Comment'
$ git push heroku master # deploy code to heroku
$ heroku ps:scale web=1  # run the app with a 1 heroku "dyno"

```

Source: [stackoverflow](https://stackoverflow.com/questions/47949173/deploy-a-python-app-to-heroku-using-conda-environments-instead-of-virtualenv)
