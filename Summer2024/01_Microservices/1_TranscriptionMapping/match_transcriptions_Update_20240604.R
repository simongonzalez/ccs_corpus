library(officer)
library(textreadr)
library(tidyverse)
library(tidytext)
library(srt)
library(rPraat)
library(quanteda)
library("quanteda.textstats")
library(readtext)
library(readr)
library(readxl)

#duration files
fls_df <- read.csv('path/to/DurationFiles_Updated_20240604.csv')

fls_doc = list.files('path/to/AllCorrectedOriginals',
                     pattern = '.doc', full.names = TRUE)

fls_son = list.files('path/to/Transcripts_Sonix_5sec', pattern = '.srt', full.names = TRUE)

dfall_son = data.frame(srt = fls_son, file_srt = basename(fls_son), 
                       name = gsub('\\.mp3\\.srt', '', basename(fls_son)))

dfall = data.frame(doc = fls_doc, file_doc = basename(fls_doc), 
                   name = gsub('\\.doc', '', basename(fls_doc))) %>%
  left_join(dfall_son, by = 'name') %>%
  left_join(fls_df, by = 'name') %>%
  drop_na(srt)


for(filei in 1:nrow(dfall)){
  
  print(filei)
  
  infile_path = dfall$doc[filei]
  
  #...............................................................................
  #Read in manual transcription
  df_doc <- data.frame(raw = str_squish(unlist(strsplit(read_doc(infile_path), '\n')))) %>%
    mutate(speaker = case_when(
      grepl('^Habl|^HABL|^I\\.|^I\\:|^O\\:', raw) ~ 'Hablante',
      grepl('^Enc\\.\\:|^Enc\\.[0-9]\\:|^E\\.|^E\\[0-9]\\:|^AUX[0-9]\\:', raw) ~ 'Entrevistador',
      TRUE ~ NA
    )) %>%
    fill(speaker, .direction = 'down') %>%
    mutate(text = str_squish(gsub('Habl\\.|Habl\\.\\:|Habl\\:|Enc\\.|Enc\\.\\:|Enc\\.[0-9]\\:|HABL\\:|I\\.\\:|E\\.\\:|E[0-9]\\:|AUX1\\:|O\\:', '', raw))) %>%
    drop_na(speaker) %>%
    mutate(text = str_remove(text, "\\s*\\([^\\)]+\\)|\\s*\\[[^\\]]+\\]|\\s*<[^>]+>")) %>%
    mutate(text = tolower(str_squish(gsub('\\/|\\,|\\.|\\?|\\¿|\\.\\.\\.|\\.\\.|!|¡|"', '', text)))) %>%
    mutate(text = gsub(" pa' ", ' para ', text)) %>%
    mutate(text = gsub("nadien", 'nadie', text)) %>%
    mutate(line_number_n = 1:n())
  
  #...............................................................................
  #...............................................................................
  #Get all ngrams from the manual transcription
  df_doc_ngram_10 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 10, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_9 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 9, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_8 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 8, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_7 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 7, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_6 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 6, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_5 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 5, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_4 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 4, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_3 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 3, drop = FALSE) %>%
    drop_na(ngram_text)
  
  df_doc_ngram_2 = df_doc %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 2, drop = FALSE) %>%
    drop_na(ngram_text)
  
  #...............................................................................
  #Read in Sonix transcription
  dfson = read_srt(dfall$srt[filei], collapse = "\n") %>%
    mutate(text = tolower(str_squish(gsub('SPEAKER[0-9]\\:|\\,|\\.|\\?|\\¿|\\.\\.\\.|\\.\\.|!|¡', '', subtitle)))) %>%
    mutate(word_n = lengths(gregexpr("\\W+", text)) + 1) %>%
    filter(word_n >= 5) %>%
    group_by(text) %>%
    mutate(text_unique = n()) %>%
    ungroup() %>%
    filter(text_unique == 1)
  
  dfson_ngram_10 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 10, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_9 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 9, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_8 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 8, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_7 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 7, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_6 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 6, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_5 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 5, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_4 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 4, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_3 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 3, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  dfson_ngram_2 = dfson %>%
    unnest_tokens(ngram_text, text, token = "ngrams", n = 2, drop = FALSE) %>%
    drop_na(ngram_text) %>%
    dplyr::select(n, start, end, ngram_text)
  
  #...............................................................................
  #Match manual and automatic transcription
  df_matched_10 = df_doc_ngram_10 %>%
    left_join(dfson_ngram_10, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 10)
  
  df_matched_9 = df_doc_ngram_9 %>%
    left_join(dfson_ngram_9, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 9)
  
  df_matched_8 = df_doc_ngram_8 %>%
    left_join(dfson_ngram_8, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 8)
  
  df_matched_7 = df_doc_ngram_7 %>%
    left_join(dfson_ngram_7, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 7)
  
  df_matched_6 = df_doc_ngram_6 %>%
    left_join(dfson_ngram_6, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 6)
  
  df_matched_5 = df_doc_ngram_5 %>%
    left_join(dfson_ngram_5, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 5)
  
  df_matched_4 = df_doc_ngram_4 %>%
    left_join(dfson_ngram_4, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 4)
  
  df_matched_3 = df_doc_ngram_3 %>%
    left_join(dfson_ngram_3, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 3)
  
  df_matched_2 = df_doc_ngram_2 %>%
    left_join(dfson_ngram_2, by = 'ngram_text', relationship = "many-to-many") %>%
    drop_na(n) %>%
    distinct(line_number_n, .keep_all = TRUE) %>%
    mutate(ngram_number = 2)
  
  #find matches between 10 and 6
  matches_from_9 = setdiff(df_matched_9$line_number_n, df_matched_10$line_number_n)
  df_matched_all = bind_rows(df_matched_10, df_matched_9 %>% filter(line_number_n %in% matches_from_9)) %>%
    arrange(line_number_n)
  
  matches_from_8 = setdiff(df_matched_8$line_number_n, df_matched_all$line_number_n)
  df_matched_all = bind_rows(df_matched_all, df_matched_8 %>% filter(line_number_n %in% matches_from_8)) %>%
    arrange(line_number_n)
  
  matches_from_7 = setdiff(df_matched_7$line_number_n, df_matched_all$line_number_n)
  df_matched_all = bind_rows(df_matched_all, df_matched_7 %>% filter(line_number_n %in% matches_from_7)) %>%
    arrange(line_number_n)
  
  matches_from_6 = setdiff(df_matched_6$line_number_n, df_matched_all$line_number_n)
  df_matched_all = bind_rows(df_matched_all, df_matched_6 %>% filter(line_number_n %in% matches_from_6)) %>%
    arrange(line_number_n)
  
  matches_from_5 = setdiff(df_matched_5$line_number_n, df_matched_all$line_number_n)
  df_matched_all = bind_rows(df_matched_all, df_matched_5 %>% filter(line_number_n %in% matches_from_5)) %>%
    arrange(line_number_n)
  
  matches_from_4 = setdiff(df_matched_4$line_number_n, df_matched_all$line_number_n)
  df_matched_all = bind_rows(df_matched_all, df_matched_4 %>% filter(line_number_n %in% matches_from_4)) %>%
    arrange(line_number_n)
  
  #
  df_speaker = df_matched_all %>% filter(speaker == 'Hablante') %>%
    filter(!(end < start))
  
  for(timerep in 1:10){
    for(rowi in nrow(df_speaker):2){
      tmp_start = df_speaker$start[rowi]
      tmp_start_previous = df_speaker$start[rowi-1]
      
      if(tmp_start < tmp_start_previous){
        df_speaker = df_speaker %>% slice(-rowi)
      }
    }
  }
  
  
  df_speaker = df_speaker %>% distinct(start, .keep_all = TRUE)
  
  for(rowi in 1:(nrow(df_speaker)-1)){
    tmp_end = df_speaker$end[rowi]
    tmp_start_next = df_speaker$start[rowi+1]
    
    if(tmp_end > tmp_start_next){
      df_speaker$end[rowi] = tmp_start_next
    }
  }
  
  tmpspeakerdf <- df_speaker %>%
      arrange(start) %>%
      filter(end > start)
  
  #CA1HA_87
  tgdur = dfall$dur[filei]
  
  #Creates TextGrid
  tg <- rPraat::tg.createNewTextGrid(0, tgdur)
  
  #Add anchor times
  interval_cntr <- 1
  
  #Insert Sonix Transcription
  tg <- tg.insertNewIntervalTier(tg = tg, newTierName = 'Speaker', 
                                 newInd = interval_cntr)
  interval_cntr <- interval_cntr + 1
  
  
  
  for(interval_i in 1:nrow(tmpspeakerdf)){
    tg <- tg.insertInterval(tg = tg, tierInd = 'Speaker',
                            tStart = tmpspeakerdf$start[interval_i], 
                            tEnd = tmpspeakerdf$end[interval_i], 
                            label = as.character(tmpspeakerdf$text[interval_i]))
  }
  
  #Saves the TextGrid
  save_name <- paste0('./XX_TG_Matched/', dfall$name[filei], '.TextGrid')
  tg.write(tg = tg, fileNameTextGrid = save_name)
}



