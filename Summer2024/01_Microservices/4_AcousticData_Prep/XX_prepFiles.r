#https://cran.r-project.org/web/packages/udpipe/vignettes/udpipe-annotation.html

library(rPraat)
library(dplyr)
library(stringr)
library(ggplot2)
library(plotly)
library(readr)
library(parallel)
library(foreach)
library(doParallel)
library(tictoc)
library(tidyverse)
library(emuR)
library(udpipe)
library(vowels)
library(phonfieldwork)
library(readtextgrid)

`%nin%` <- Negate(`%in%`)

source('./functions/sd_calculation_function.R')
source('./functions/findSegments.R')
source('./functions/findTranscription.R')
source('./functions/findWords.R')
source('./functions/get_name_parameter.R')

fls_inputs = list.files('/Users/calejohnstone/Documents/wk/ESP/2024/CCS/CCSFiles/forcedalignment/ccs_test2', full.names = T, pattern = '.TextGrid')

fls_outputs <- list.files('/Users/calejohnstone/Documents/wk/ESP/2024/CCS/CCSFiles/forcedalignment/ccs_test2_output', full.names = T, pattern = '.TextGrid')

df_fls_in = data.frame(files_in_loc = fls_inputs, files = basename(fls_inputs))

df_fls = data.frame(files_out_loc = fls_outputs, files = basename(fls_outputs)) %>%
  left_join(df_fls_in, by = c('files'))

cntr <- 1

for(i in 1:nrow(df_fls)){
  
  print(paste0(cntr, '/', nrow(df_fls)))
  #tg <- tg.read(i, encoding = as.character(guess_encoding(i)$encoding[1]))
  
  tg_input = tg.read(df_fls$files_in_loc[i], 
                     encoding = as.character(guess_encoding(df_fls$files_in_loc[i])$encoding[1]))
  
  df_in = data.frame(tStart = tg_input$Text$t1, tEnd = tg_input$Text$t2, label = tg_input$Text$label) %>%
    filter(label != '')
  
  tg_output <- tg.read(df_fls$files_out_loc[i], 
                encoding = as.character(guess_encoding(df_fls$files_out_loc[i])$encoding[1]))
  
  #Add text tier
  interval_cntr <- 1
  
  #Insert Transcriptions
  tg_output <- tg.insertNewIntervalTier(tg = tg_output, newTierName = 'Text', 
                                 newInd = interval_cntr)
  
  for(interval_i in 1:nrow(df_in)){
    tg_output <- tg.insertInterval(tg = tg_output, tierInd = 'Text',
                            tStart = df_in$tStart[interval_i], 
                            tEnd = df_in$tEnd[interval_i], 
                            label = as.character(df_in$label[interval_i]))
  }
  
  fullname <- gsub('.TextGrid', '', basename(df_fls$files[i]))
  #namesSplit <- unlist(strsplit(fullname, '_'))
  
  #tmpRegion <- 1
  #tmpGender <- 1
  #tmpName <- 1
  #tmpRec <- 1
  
  tg = tg_output
  
  textTierLabel <- names(tg)[1]
  wordTierLabel <- names(tg)[2]
  segmentTierLabel <- names(tg)[3]

  tmpSegsLocs <- which(!tg[[segmentTierLabel]]$label %in% c('', 'sp', 'sil'))
  tmpSegs <- str_squish(tg[[segmentTierLabel]]$label[tmpSegsLocs])
  tmpSegsPrev <- tg[[segmentTierLabel]]$label[tmpSegsLocs-1]
  tmpSegsFoll <- tg[[segmentTierLabel]]$label[tmpSegsLocs+1]
  tmpSegsOnset <- tg[[segmentTierLabel]]$t1[tmpSegsLocs]
  tmpSegsOffset <- tg[[segmentTierLabel]]$t2[tmpSegsLocs]
  tmpSegsDur <- tmpSegsOffset - tmpSegsOnset
  
  #get texts
  #=======================================================================
  textlabel = NULL
  textOnset = NULL
  textOffset = NULL
  textMid = NULL
  textDur = NULL
  textLoc <- NULL
  
  for(j in 1:length(tmpSegsLocs)){
    textlabel[j] = str_squish(findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = textTierLabel)[[1]])
    textOnset[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = textTierLabel)[[2]]
    textOffset[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = textTierLabel)[[3]]
    textMid[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = textTierLabel)[[4]]
    textDur[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = textTierLabel)[[5]]
    textLoc[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = textTierLabel)[[6]]
  }
  
  #get previous words
  prevTexts <- tg[[textTierLabel]]$label[textLoc-1]
  
  #get following words
  follTexts <- tg[[textTierLabel]]$label[textLoc+1]
  
  #get words
  #=======================================================================
  wordlabel = NULL
  wordOnset = NULL
  wordOffset = NULL
  wordMid = NULL
  wordDur = NULL
  wordLoc <- NULL
  
  for(j in 1:length(tmpSegsLocs)){
    wordlabel[j] = str_squish(findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = wordTierLabel)[[1]])
    wordOnset[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = wordTierLabel)[[2]]
    wordOffset[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = wordTierLabel)[[3]]
    wordMid[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = wordTierLabel)[[4]]
    wordDur[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = wordTierLabel)[[5]]
    wordLoc[j] = findWords(data = tg, segNumber = tmpSegsLocs[j], tierLabel = segmentTierLabel, wordTier = wordTierLabel)[[6]]
  }
  
  #get previous words
  prevWords <- tg[[wordTierLabel]]$label[wordLoc-1]
  
  #get following words
  follWords <- tg[[wordTierLabel]]$label[wordLoc+1]
  
  
  tmpdf <- data.frame(speaker = fullname,
                      #name = tmpName, gender = tmpGender, region = tmpRegion, rec = tmpRec,
                      previous = tmpSegsPrev, segment = tmpSegs, following = tmpSegsFoll, 
                      onset = tmpSegsOnset, offset = tmpSegsOffset, dur = tmpSegsDur,
                      previousWord = prevWords, word = wordlabel, followingWord = follWords, 
                      wordOnset, wordOffset, wordDur,
                      previousText = prevTexts, text = textlabel, followingText = follTexts, 
                      textOnset, textOffset, textDur)
  
  if(i == 1){
    dat <- tmpdf
  }else{
    dat <- rbind(dat, tmpdf)
  }
  
  cntr = cntr + 1
  
}

#Add descriptors
df = dat
#identify the position of the segment in the word
df$position <- ifelse(df$onset == df$wordOnset, 'initial', 'medial')
df[df$offset == df$wordOffset, 'position'] <- 'final'
df[df$onset == df$wordOnset & df$offset == df$wordOffset, 'position'] <- 'both'

df$wordposition <- ifelse(df$wordOnset == df$textOnset, 'initial', 'medial')
df[df$wordOffset == df$textOffset, 'wordposition'] <- 'final'
df[df$wordOnset == df$textOnset & df$wordOffset == df$textOffset, 'wordposition'] <- 'both'

#get all segments
all_segments = c(sort(unique(df$previous)), sort(unique(df$segment)), sort(unique(df$following)))
paste(sort(unique(all_segments)), collapse = ';')

df %>% filter(segment %in% unlist(strsplit('x', ';'))) %>% View()

segment_mapping = setNames(unlist(strsplit("H;d;D;G;J;Y;y;N;M;R;t;C;B;T", ';')),
                           unlist(strsplit("ç;d̪;ð;ɣ;ʝ;ɟʝ;ʎ;ɲ;ŋ;ɾ;t̪;tʃ;β;θ", ';')))

df$segments = segment_mapping[as.character(df$segment)]
df$previous_segs = segment_mapping[as.character(df$previous)]
df$following_segs = segment_mapping[as.character(df$following)]

df = df %>%
  mutate(segments = if_else(is.na(segments), segment, segments)) %>%
  mutate(previous_segs = if_else(is.na(previous_segs), previous, previous_segs)) %>%
  mutate(following_segs = if_else(is.na(following_segs), following, following_segs))

all_segments_mapped = c(sort(unique(df$previous_segs)), sort(unique(df$segments)), sort(unique(df$following_segs)))
paste(sort(unique(all_segments_mapped)), collapse = ';')

original_segments = unlist(strsplit(";a;b;β;c;tʃ;d̪;ð;e;f;ɣ;ɡ;ç;i;j;k;l;ʝ;m;ɲ;n;ŋ;o;p;r;ɾ;s;spn;t̪;θ;u;w;x;ɟʝ;ʎ", ';'))
source_segments = unlist(strsplit("blank;a;b;B;q;C;d;D;e;f;G;ɡ;H;i;j;k;l;J;m;M;n;N;o;p;r;R;s;spn;t;T;u;w;x;Y;y", ';'))
phonological_segments = unlist(strsplit("blank;a;b;b;k;C;d;d;e;f;g;ɡ;h;i;i;k;l;J;m;M;n;n;o;p;r;R;s;spn;t;s;u;u;h;J;J", ';'))
type_segments = unlist(strsplit("blank;vowel;consonant;consonant;consonant;consonant;consonant;consonant;vowel;consonant;consonant;consonant;consonant;vowel;semivowel;consonant;consonant;consonant;consonant;consonant;consonant;consonant;vowel;consonant;consonant;consonant;consonant;spn;consonant;consonant;vowel;semivowel;consonant;consonant;consonant", ';'))
manner_segments = unlist(strsplit("blank;low;stop;approximant;stop;affricate;stop;approximant;mid;fricative;approximant;stop;fricative;high;approximant;stop;lateral;affricate;nasal;nasal;nasal;nasal;mid;stop;trill;tap;fricative;spn;stop;fricative;high;approximant;fricative;affricate;affricate", ';'))
place_segments = unlist(strsplit("blank;central;bilabial;bilabial;velar;palatal;dental;dental;front;labiodental;velar;velar;glottal;front;palatal;velar;alveolar;palatal;bilabial;palatal;alveolar;velar;back;bilabial;alveolar;alveolar;alveolar;spn;dental;dental;back;labiovelar;glottal;palatal;palatal", ';'))
majorplace_segments = unlist(strsplit("blank;CENTRAL;LABIAL;LABIAL;DORSAL;CORONAL;CORONAL;CORONAL;FRONT;LABIAL;DORSAL;DORSAL;PHARYNGEAL;FRONT;CORONAL;DORSAL;CORONAL;CORONAL;LABIAL;CORONAL;CORONAL;DORSAL;BACK;LABIAL;CORONAL;CORONAL;CORONAL;spn;CORONAL;CORONAL;BACK;LABIAL;PHARYNGEAL;CORONAL;CORONAL", ';'))
voicing_segments = unlist(strsplit("blank;voiced;voiced;voiced;unvoiced;unvoiced;voiced;voiced;voiced;unvoiced;voiced;voiced;unvoiced;voiced;voiced;unvoiced;voiced;voiced;voiced;voiced;voiced;voiced;voiced;unvoiced;unvoiced;unvoiced;unvoiced;spn;unvoiced;unvoiced;voiced;voiced;unvoiced;voiced;voiced", ';'))
rounding_segments = unlist(strsplit("blank;unrounded;NotAp;NotAp;NotAp;NotAp;NotAp;NotAp;unrounded;NotAp;NotAp;NotAp;NotAp;unrounded;unrounded;NotAp;NotAp;NotAp;NotAp;NotAp;NotAp;NotAp;rounded;NotAp;NotAp;NotAp;NotAp;spn;NotAp;NotAp;rounded;rounded;NotAp;NotAp;NotAp", ';'))

phonologicalFeatures = data.frame(original_segments, source_segments, phonological_segments, type_segments, manner_segments, place_segments, majorplace_segments, voicing_segments, rounding_segments)

phonologicalFeatures_previous = phonologicalFeatures
names(phonologicalFeatures_previous) = paste0('previous_', names(phonologicalFeatures_previous))

phonologicalFeatures_following = phonologicalFeatures
names(phonologicalFeatures_following) = paste0('following_', names(phonologicalFeatures_following))

names(phonologicalFeatures) = paste0('segment_', names(phonologicalFeatures))

#create mappings
df_phon = df %>%
  left_join(phonologicalFeatures_previous %>% 
              rename(previous = previous_original_segments), by = 'previous') %>%
  left_join(phonologicalFeatures %>% 
              rename(segment = segment_original_segments), by = 'segment') %>%
  left_join(phonologicalFeatures_following %>% 
              rename(following = following_original_segments), by = 'following') %>%
  mutate(time = onset + ((offset - onset)/2), row_index = 1:n())

write.csv(df_phon, 'df_phon.csv', row.names = FALSE)

dat_append <- df_phon %>%
  dplyr::select(speaker, time, row_index) %>%
  mutate(
    formant_1 = 0, 
    formant_2 = 0, 
    formant_3 = 0, 
    pitchValue = 0, 
    intensityValue = 0
  )

write.csv(dat_append, 'dat_append.csv', row.names = FALSE, quote = F)

write.csv(dat, 'icphs23_new_Second_preped.csv', row.names = F, quote = F)
write.csv(dat_append, 'icphs23_new_Second_append.csv', row.names = F, quote = F)

#Add description of the type of vowels, diphthongs, rising, falling
#Speech rate
#Stress information
#Add POS information

df_phon %>%
  group_by(segment_type_segments) %>%
  count() %>%
  View()

df_phon %>%
  filter(segment_type_segments == 'vowel') %>%
  group_by(segment) %>%
  summarise(dur = mean((dur*1000), na.rm = TRUE)) %>%
  ggplot(aes(segment, dur)) +
  geom_bar(stat='identity')


df <- df %>%
  mutate(type = case_when(
    segment %in% unlist(strsplit('a A aj AJ aw AW e E ej EJ ew EW i I ja JA je JE jo JO ju JU o O oj OJ u U uj UJ wa WA waj WAJ we WE wi WI wo WO', ' ')) ~ 'vowel',
    segment %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
    segment %in% unlist(strsplit('b C d g k p t m n N f s h S B D G l r R Y', ' ')) ~ 'semivowel',
    TRUE ~ 'numeric'
  ))

df <- df %>%
  mutate(
    prevManner = case_when(
      previous %in% unlist(strsplit('b C d g k p t', ' ')) ~ 'stop',
      previous %in% unlist(strsplit('m n N', ' ')) ~ 'nasal',
      previous %in% unlist(strsplit('f s h S B D G', ' ')) ~ 'fricative',
      previous %in% unlist(strsplit('l r R Y', ' ')) ~ 'liquid',
      previous %in% c('', ' ') ~ 'blank',
      previous %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      previous %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    manner = case_when(
      segment %in% unlist(strsplit('b C d g k p t', ' ')) ~ 'stop',
      segment %in% unlist(strsplit('m n N', ' ')) ~ 'nasal',
      segment %in% unlist(strsplit('f s h S B D G', ' ')) ~ 'fricative',
      segment %in% unlist(strsplit('l r R Y', ' ')) ~ 'liquid',
      segment %in% c('', ' ') ~ 'blank',
      segment %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      segment %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    follManner = case_when(
      following %in% unlist(strsplit('b C d g k p t', ' ')) ~ 'stop',
      following %in% unlist(strsplit('m n N', ' ')) ~ 'nasal',
      following %in% unlist(strsplit('f s h S B D G', ' ')) ~ 'fricative',
      following %in% unlist(strsplit('l r R Y', ' ')) ~ 'liquid',
      following %in% c('', ' ') ~ 'blank',
      following %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      following %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    
    prevMannerMaj = case_when(
      previous %in% unlist(strsplit('b C d g k p t f s h S B D G', ' ')) ~ 'obstruent',
      previous %in% unlist(strsplit('m n N l r R Y', ' ')) ~ 'sonorant',
      previous %in% c('', ' ') ~ 'blank',
      previous %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      previous %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    mannerMaj = case_when(
      segment %in% unlist(strsplit('b C d g k p t f s h S B D G', ' ')) ~ 'obstruent',
      segment %in% unlist(strsplit('m n N l r R Y', ' ')) ~ 'sonorant',
      segment %in% c('', ' ') ~ 'blank',
      segment %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      segment %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    follMannerMaj = case_when(
      following %in% unlist(strsplit('b C d g k p t f s h S B D G', ' ')) ~ 'obstruent',
      following %in% unlist(strsplit('m n N l r R Y', ' ')) ~ 'sonorant',
      following %in% c('', ' ') ~ 'blank',
      following %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      following %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    
    
    prevPlaceMaj = case_when(
      previous %in% unlist(strsplit('b m p f B', ' ')) ~ 'labial',
      previous %in% unlist(strsplit('C d l n N r R s S t Y D', ' ')) ~ 'coronal',
      previous %in% unlist(strsplit('g k h G', ' ')) ~ 'dorsal',
      previous %in% c('', ' ') ~ 'blank',
      previous %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      previous %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    placeMaj = case_when(
      segment %in% unlist(strsplit('b m p f B', ' ')) ~ 'labial',
      segment %in% unlist(strsplit('C d l n N r R s S t Y D', ' ')) ~ 'coronal',
      segment %in% unlist(strsplit('g k h G', ' ')) ~ 'dorsal',
      segment %in% c('', ' ') ~ 'blank',
      segment %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      segment %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    follPlaceMaj = case_when(
      following %in% unlist(strsplit('b m p f B', ' ')) ~ 'labial',
      following %in% unlist(strsplit('C d l n N r R s S t Y D', ' ')) ~ 'coronal',
      following %in% unlist(strsplit('g k h G', ' ')) ~ 'dorsal',
      following %in% c('', ' ') ~ 'blank',
      following %in% unlist(strsplit('a A e E i I o O u U', ' ')) ~ 'vowel',
      following %in% unlist(strsplit('j J w W', ' ')) ~ 'semivowel',
      TRUE ~ 'numeric'),
    
    prevVoicing = case_when(
      previous %in% unlist(strsplit('a A aj AJ aw AW b d e E ej EJ ew EW g i I ja JA je JE jo JO ju JU l m n N o O oj OJ r R u U uj UJ wa WA waj WAJ we WE wi WI wo WO Y B D G J j w W', ' ')) ~ 'voiced',
      previous %in% unlist(strsplit('C f k p s S t h', ' ')) ~ 'voiceless',
      previous %in% c('', ' ') ~ 'blank'
    ),
    voicing = case_when(
      segment %in% unlist(strsplit('a A aj AJ aw AW b d e E ej EJ ew EW g i I ja JA je JE jo JO ju JU l m n N o O oj OJ r R u U uj UJ wa WA waj WAJ we WE wi WI wo WO Y B D G J j w W', ' ')) ~ 'voiced',
      segment %in% unlist(strsplit('C f k p s S t h', ' ')) ~ 'voiceless',
      segment %in% c('', ' ') ~ 'blank'
    ),
    follVoicing = case_when(
      following %in% unlist(strsplit('a A aj AJ aw AW b d e E ej EJ ew EW g i I ja JA je JE jo JO ju JU l m n N o O oj OJ r R u U uj UJ wa WA waj WAJ we WE wi WI wo WO Y B D G J j w W', ' ')) ~ 'voiced',
      following %in% unlist(strsplit('C f k p s S t h', ' ')) ~ 'voiceless',
      following %in% c('', ' ') ~ 'blank'
    )
  )

# df %>%
#   filter(is.na(follVoicing)) %>%
#   distinct(following, .keep_all = T) %>%
#   View()

df$mid <- df$onset + ((df$offset - df$onset)/2)
df$wordMid <- df$wordOnset + ((df$wordOffset - df$wordOnset)/2)

write.csv(df, 'icphs23_new_Second.csv', row.names = F)

#add phonInformation
pronDict_stress <- read.csv('pronDict_stress.csv') %>%
  mutate(wordlower = tolower(allwords))

dfnum <- df %>%
  filter(grepl('^[0-9]', word)) %>%
  group_by(fullname, wordOnset) %>%
  mutate(group_id = cur_group_id())

for(i in sort(unique(dfnum$group_id))){
  tmpdf <- dfnum %>%
    filter(group_id == i)
  
  tmpPron <- tmpdf$segment
  vowelCount <- length(which(tmpPron %in% c('a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U')))
  
  pronDict_stress[pronDict_stress$wordlower == unique(tmpdf$word), 'entries'] <- paste(tmpPron, collapse = ' ')
  pronDict_stress[pronDict_stress$wordlower == unique(tmpdf$word), 'vowelCount'] <- vowelCount
}

df <- df %>%
  mutate(wordlower = tolower(word)) %>%
  left_join(pronDict_stress, by = 'wordlower') %>%
  filter(vowelCount != 0)

write.csv(df, 'icphs23_new_Second.csv', row.names = F)
#========================================
#Check words and counts
#add word id
df <- df %>%
  group_by(fullname, wordOnset) %>%
  mutate(group_id = cur_group_id())

#identify contiguous words 
for(i in sort(unique(df$fullname))){
  print(i)
  tmpdf <- df %>% filter(fullname == i)
  
  tmpdf$tmpFollowing <- c(tmpdf$onset[2:nrow(tmpdf)], 10000000000000)
  tmpdf$contiguous <- ifelse(tmpdf$offset == tmpdf$tmpFollowing, 1, 0)
  tmpdf$tmpFollowingSeg <- c(tmpdf$following[2:nrow(tmpdf)], '')
  
  tmpdf <- tmpdf %>%
    mutate(diph = case_when(
      contiguous == 1 & segment %in% c('a', 'e', 'o', 'A', 'E', 'O') & following %in% c('j', 'u', 'J', 'W') ~ 'Rising',
      contiguous == 1 & segment %in% c('j', 'u', 'J', 'W') & following %in% c('a', 'e', 'o', 'A', 'E', 'O') ~ 'Falling',
      contiguous == 1 & segment %in% c('a', 'e', 'o', 'A', 'E', 'O') & following %in% c('a', 'e', 'o', 'A', 'E', 'O') ~ 'VV',
      contiguous == 1 & segment %in% c('j', 'u', 'J', 'W') & following %in% c('j', 'u', 'J', 'W') ~ 'Clash',
      TRUE ~ 'Other'
      
    )) %>%
    mutate(stress = case_when(
      segment %in% c('A', 'E', 'O', 'I', 'U', 'J', 'W') ~ 'Stressed',
      segment %in% c('a', 'e', 'o', 'i', 'u', 'j', 'w') ~ 'Unstressed',
      TRUE ~ 'Other'
    ))
  
  tmpdf$tmpFollowingWord <- c(tmpdf$wordOnset[2:nrow(tmpdf)], 10000000000000)
  tmpdf$contiguousWord <- ifelse((tmpdf$tmpFollowingWord - tmpdf$wordOffset) <= 0.6 , 1, 0)
  tmpdf$contiguousWordRate <- ifelse((tmpdf$tmpFollowingWord - tmpdf$wordOffset) <= 2 , 1, 0)
  
  tmpdf <- tmpdf %>%
    group_by(wordOnset) %>%
    mutate(word_id = cur_group_id())

  #Identify contiguous words
  for(iii in sort(unique(tmpdf$word_id))){
    tmpdfiii <- tmpdf %>%
      filter(word_id == iii) %>%
      .$contiguousWord %>% as.numeric() %>% unique()
    
    if(length(which(tmpdfiii == 0)) != 1){
      nextWord <- tmpdf %>%
        filter(word_id == (iii + 1)) %>%
        .$word %>% as.character() %>% unique()
      
      currentWord <- tmpdf %>%
        filter(word_id == (iii)) %>%
        .$word %>% as.character() %>% unique()

      tmpdf[tmpdf$word_id == (iii),'followingWord'] <- nextWord
      tmpdf[tmpdf$word_id == (iii + 1),'previousWord'] <- currentWord
    }
 
  }
  
  if(i == sort(unique(df$fullname))[1]){
    dffullall <- tmpdf
  }else{
    dffullall <- rbind(dffullall, tmpdf)
  }
  
}

write.csv(dffullall, 'icphs23_new_Second.csv', row.names = F)

dffullall <- read.csv('icphs23_new_Second.csv')

dffullall$nins <- 1:nrow(dffullall)
dffullall$speechRate <- 0

dffullall$contiguousWordRate <- ifelse((dffullall$tmpFollowingWord - dffullall$wordOffset) <= 2 , 1, 0)


for(i in sort(unique(dffullall$fullname))){
  print(i)
  cntr <- 1
  tmpdf <- dffullall %>% filter(fullname == i)

  tokenLocation <- which(tmpdf$segment %in% c('a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U'))

  for(j in tokenLocation){
    word_start <- tmpdf[j,]
    word_id <- word_start$word_id

    if(word_id >= 4 & word_id < (max(tmpdf$word_id) - 3)){
      tmpcal <- tmpdf[tmpdf$word_id %in% c((word_id-3):(word_id+3)), ]

      if(length(which(tmpcal$contiguousWordRate == 0)) == 0){

        #print(cntr)
        cntr <- cntr + 1
        syllNum <- tmpcal %>%
          distinct(word, .keep_all = T) %>%
          .$vowelCount %>% sum()

        timeNum <- tmpcal %>%
          distinct(word, .keep_all = T) %>%
          mutate(worDurTemp = wordDur * 1000) %>%
          .$worDurTemp %>% sum()

        tmpRate <- syllNum / timeNum

        tmpnins <- word_start$nins

        dffullall[dffullall$fullname == i & dffullall$nins == tmpnins, 'speechRate'] <- tmpRate
      }

    }

  }
}

dffullall$index <- 1:nrow(dffullall)

write.csv(dffullall, 'icphs23_new_Second_sr.csv', row.names = F)

#all elements = F1, F2, F3, pitch, intensity
df <- dffullall

# time_df <- as.data.frame(matrix(nrow = nrow(df), ncol = 57))
# names(time_df) <- c('index', 'midTime', paste0('f1_', 1:11), paste0('f2_', 1:11), paste0('f3_', 1:11), paste0('pitch_', 1:11), paste0('intensity_', 1:11))
# 
# newn <- 1

dfv <- df# %>% filter(segment %in% c('a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U', 'j', 'J', 'w'))

# for(i in 1:nrow(dfv)){
#   print(i)
#   tmpSeq <- seq(dfv$onset[i], dfv$offset[i], length.out = 1)
#   tmpline <- dfv[i,] %>%
#     slice(rep(1:n(), each = newn)) %>%
#     mutate(token = i,
#            perc = 1:1,
#            time = tmpSeq
#     )
#   
#   if(i == 1){
#     dat <- tmpline
#   }else{
#     dat <- rbind(dat, tmpline)
#   }
# }

dat <- dfv %>%
  mutate(time = onset + ((offset - onset)/2)) %>%
  mutate(row_index = 1:n()) %>%
  mutate(feats = gsub(',', ';', feats))

# dat_tmp <- dat
# 
# dat <- dat %>%
#   select(-one_of(c('formant_1', 'formant_2', 'formant_3', 'pitchValue', 'intensityValue',
#                    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
#                    'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12'
#                    ))) %>%
#   mutate(row_index = 1:n())

dat_append <- dat %>%
  dplyr::select(speaker, time, row_index) %>%
  mutate(
    formant_1 = 0, 
    formant_2 = 0, 
    formant_3 = 0, 
    pitchValue = 0, 
    intensityValue = 0,
    mfcc_1 = 0,
    mfcc_2 = 0,
    mfcc_3 = 0,
    mfcc_4 = 0,
    mfcc_5 = 0,
    mfcc_6 = 0,
    mfcc_7 = 0,
    mfcc_8 = 0,
    mfcc_9 = 0,
    mfcc_10 = 0,
    mfcc_11 = 0,
    mfcc_12 = 0
  )

write.csv(dat, 'icphs23_new_Second_preped.csv', row.names = F, quote = F)
write.csv(dat_append, 'icphs23_new_Second_append.csv', row.names = F, quote = F)

dat_sr <- dat %>%
  filter(speechRate != 0)

dat_append_sr <- dat_sr %>%
  dplyr::select(speaker, time, row_index) %>%
  mutate(
    formant_1 = 0, 
    formant_2 = 0, 
    formant_3 = 0, 
    pitchValue = 0, 
    intensityValue = 0,
    mfcc_1 = 0,
    mfcc_2 = 0,
    mfcc_3 = 0,
    mfcc_4 = 0,
    mfcc_5 = 0,
    mfcc_6 = 0,
    mfcc_7 = 0,
    mfcc_8 = 0,
    mfcc_9 = 0,
    mfcc_10 = 0,
    mfcc_11 = 0,
    mfcc_12 = 0
  )


write.csv(dat_sr, 'icphs23_new_Second_preped_dat_sr.csv', row.names = F, quote = F)
write.csv(dat_append_sr, 'icphs23_new_Second_append_dat_append_sr.csv', row.names = F, quote = F)

dat_append_sr %>%
  filter(speaker == 'CA2MB04') %>%
  View()

sort(unique(dat_append_sr$speaker))

dat_sr %>%
  group_by(segment) %>%
  count() %>%
  View()

# save(dat, file = 'hls_20220812.RData')
# save(dat_append, file = 'hls_20220812_append.RData')
