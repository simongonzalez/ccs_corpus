library(tidyverse)
library(rPraat)
library(chron)
library(lubridate)
library(tuneR)
library(readr)
library(textreadr)
library(stringdist)
library(doc2concrete)
library(textclean)
library(textreg)
library(tm)
library(diffr)
library(data.table)
library(zoo)
library("textgRid")
library(readtextgrid)

options(digits.secs=3)

fls_tg = list.files('./XX_TGS_manualFix', full.names = TRUE, pattern = '.TextGrid')

for(i in 1:fls_tg){
  tgin = readtextgrid::read_textgrid(fls_tg[i])
  
  get_times = tgin %>%
    filter(tier_name == 'Check' & text == 'Check' & annotation_num == 2)
  
  get_begin = get_times$xmin
  get_end = get_times$xmax
  
  tgdur = unique(tgin$tier_xmax)
  
  tg_filtered = tgin %>%
    filter(tier_name == 'Raw Text') %>%
    filter(xmin >= get_begin) %>%
    filter(xmax <= get_end)
  
  #Creates TextGrid
  tg <- rPraat::tg.createNewTextGrid(0, tgdur)
  
  #Add anchor times
  interval_cntr <- 1
  
  #Insert Sonix Transcription
  tg <- tg.insertNewIntervalTier(tg = tg, newTierName = 'Text', 
                                 newInd = interval_cntr)
  
  for(interval_i in 1:nrow(tg_filtered)){
    tg <- tg.insertInterval(tg = tg, tierInd = 'Text',
                            tStart = tg_filtered$xmin[interval_i], 
                            tEnd = tg_filtered$xmax[interval_i], 
                            label = as.character(tg_filtered$text[interval_i]))
  }
  
  #Saves the TextGrid
  save_name <- paste0('./01_TGS/', basename(fls_tg[i]))
  tg.write(tg = tg, fileNameTextGrid = save_name)
  
}
