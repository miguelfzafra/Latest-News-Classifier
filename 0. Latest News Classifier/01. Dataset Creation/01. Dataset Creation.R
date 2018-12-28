#################################################################################################
# 01. DATASET CREATION

# The aim of this script is to create a dataset with the following information:
#   - Name of the article
#   - Content of the article
#   - Category of the article
#################################################################################################

# Installs
# install.packages("readtext", dependencies=T)

# Imports
library(readtext)


# Cleaning environment data
rm(list = ls())

# Working directory
setwd('C:/Users/migue/Data Science/Master Data Science/KSCHOOL/9. TFM/0. Latest News Classifier/01. Dataset Creation')

# Path definition of the news archives
path <- 'C:/Users/migue/Data Science/Master Data Science/KSCHOOL/9. TFM/0. Latest News Classifier/00. Raw dataset/BBC/bbc-fulltext/bbc'

# List with the 5 categories
list_categories <- list.files(path=path)

# Save to dataset the number of files in each category folder
summary_categories <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(summary_categories) <- c('Category', 'Number_of_Files')

for (category in list_categories){
  category_path <- paste(path, category, sep='/')
  n_files <- length(list.files(path=category_path))
  
  summary_categories = rbind(summary_categories, data.frame('Category'=category, 'Number_of_Files'=n_files))
}

summary_categories

# Read every folder and create the final dataframe
df_final <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(df_final) <- c('doc_id', 'text', 'category')

for(category in list_categories){
  category_path <- paste(path, category, sep='/')

  df <- readtext(category_path)
  df["category"] = category
  
  df_final = rbind(df_final, df)
}

colnames(df_final) <- c('File_Name', 'Content', 'Category')

df_final <-
  df_final %>% 
  mutate(Complete_Filename = paste(File_Name, Category, sep='-'))

# Save dataset: .rda
save(df_final, file='Dataset.rda')

# Load dataset
load(file='Dataset.rda')

# Write csv to import to python
write.csv2(df_final,fileEncoding = 'utf8', "News_dataset.csv", row.names = FALSE)

