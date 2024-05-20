library(officer)
library(textreadr)
library(tidyverse)
library(tidytext)
library(srt)
library(rPraat)
library(quanteda)
library("quanteda.textstats")
library(stringr)
library(udpipe)

`%nin%` = Negate(`%in%`)

#UDPIPE model
mdl <- udpipe::udpipe_load_model('/Users/calejohnstone/Documents/wk/work/udpipeModels/spanish-gsd-ud-2.5-191206.udpipe')

#duration files
fls_df <- read.csv('/Users/calejohnstone/Documents/wk/ESP/2024/CCS/CCSFiles/XX_duration_files_20240324.csv')

#...............................................................................
#Read in manual transcription
df_doc <- data.frame(raw = str_squish(unlist(strsplit(read_doc('/Users/calejohnstone/Documents/wk/ESP/2024/CCS/CCSFiles/data/AllCorrectedOriginals/CA1HA_87.doc'), '\n')))) %>%
  mutate(speaker = case_when(
    grepl('^Habl', raw) ~ 'Hablante',
    grepl('^Enc\\.[0-9]:', raw) ~ 'Entrevistador',
    TRUE ~ NA
  )) %>%
  fill(speaker, .direction = 'down') %>%
  mutate(text = str_squish(gsub('Habl\\.|Habl\\.\\:|Habl\\:|Enc\\.|Enc\\.[0-9]\\:', '', raw))) %>%
  drop_na(speaker) %>%
  #delete text between [], as comments
  mutate(text = str_squish(str_replace(text, "\\s*\\[[^\\]]+\\]", ""))) %>%
  mutate(text = tolower(str_squish(gsub('\\/|"|\\.|\'', '', text)))) %>%
  filter(text %nin% c('', ' ')) %>%
  mutate(text_clean = tolower(str_squish(gsub('|\\,|\\.|\\?|\\¿|\\.\\.\\.|\\.\\.|!|¡', '', text)))) %>%
  filter(text_clean %nin% c('', ' ')) %>%
  mutate(group_turn = rleid(speaker)) %>%
  group_by(speaker) %>%
  mutate(speakerGroup_turn = rleid(group_turn)) %>%
  ungroup() %>%
  mutate(line_number_n = 1:n()) %>%
  mutate(doc_index = 1:n()) %>%
  mutate(doc_id = paste0('doc', doc_index))

#identify repetitions between turns - next turn
df_doc$repeat_nextTurn = 0
df_doc$repeat_nextTurn_word = ''

for(line_i in 1:(nrow(df_doc) - 1)){
  get_last_current_word = unlist(strsplit(df_doc$text[line_i], ' '))
  get_first_next_word = unlist(strsplit(df_doc$text[line_i + 1], ' '))
  
  if(get_last_current_word[length(get_last_current_word)] == get_first_next_word[1]){
    df_doc$repeat_nextTurn[line_i] = 1
    df_doc$repeat_nextTurn_word[line_i] = get_last_current_word[length(get_last_current_word)]
  }
}

df_ud <- udpipe::udpipe_annotate(object = mdl, x = df_doc$text)

df_ud_matched = df_ud %>% as.data.frame() %>%
  mutate(line_index = paste(doc_id, paragraph_id, sentence_id, sep = '_')) %>%
  left_join(df_doc, by = 'doc_id') %>%
  #identify questions
  mutate(is_questionTag = if_else(grepl('\\¿no\\?', text), 1, 0)) %>%
  mutate(question_structure = case_when(
    grepl('\\¿', text) & grepl('\\?', text) ~ 'Inline',
    grepl('\\¿', text) & !grepl('\\?', text) ~ 'Start',
    !grepl('\\¿', text) & grepl('\\?', text) ~ 'End',
    TRUE ~ 'Other'
  )) %>%
  #identify exclamations
  mutate(exclamation_structure = case_when(
    grepl('\\¡', text) & grepl('\\!', text) ~ 'Inline',
    grepl('\\¡', text) & !grepl('\\!', text) ~ 'Start',
    !grepl('\\¡', text) & grepl('\\!', text) ~ 'End',
    TRUE ~ 'Other'
  ))

#fix questions tagging
for(line_i in 1:(nrow(df_ud_matched) - 1)){
  if(df_ud_matched$question_structure[line_i] == 'Start' & df_ud_matched$question_structure[line_i+1] == 'Other'){
    df_ud_matched$question_structure[line_i+1] = 'Start'
  }
}

#fix exclamations tagging
for(line_i in 1:(nrow(df_ud_matched) - 1)){
  if(df_ud_matched$exclamation_structure[line_i] == 'Start' & df_ud_matched$exclamation_structure[line_i+1] == 'Other'){
    df_ud_matched$exclamation_structure[line_i+1] = 'Start'
  }
}

#Match tagging
df_ud_matched = df_ud_matched %>%
  mutate(is_question = if_else(question_structure %in% c('Inline', 'Start', 'End'), 1, 0)) %>%
  mutate(is_exclamation = if_else(exclamation_structure %in% c('Inline', 'Start', 'End'), 1, 0)) %>%
  mutate(is_statement = if_else(is_question == 0 & is_exclamation == 0, 1, 0))

df_ud_matched = df_ud_matched %>%
  mutate() %>%
  mutate(char_count = nchar(text)) %>%
  mutate(word_count = str_count(text, "\\S+")) %>%
  #overall stats
  mutate(total_Groupturns = max(group_turn)) %>%
  mutate(total_turns = n()) %>%
  mutate(total_chars = sum(char_count)) %>%
  mutate(total_words = sum(word_count)) %>%
  mutate(average_chars = mean(char_count)) %>%
  mutate(average_words = mean(word_count)) %>%
  #sentence types
  mutate(total_questionTags = sum(is_questionTag)) %>%
  mutate(total_questions = sum(is_question)) %>%
  mutate(total_exclamations = sum(is_exclamation)) %>%
  mutate(total_statements = sum(is_statement)) %>%
  #repetitions
  mutate(total_repetitions = sum(repeat_nextTurn)) %>%
  #...............................................................
  #stats by speaker
  group_by(speaker) %>%
  mutate(speaker_total_Groupturns = max(speakerGroup_turn)) %>%
  mutate(speaker_total_turns = n()) %>%
  mutate(speaker_total_chars = sum(char_count)) %>%
  mutate(speaker_total_words = sum(word_count)) %>%
  mutate(speaker_average_chars = mean(char_count)) %>%
  mutate(speaker_average_words = mean(word_count)) %>%
  #sentence types
  mutate(speaker_total_questionTags = sum(is_questionTag)) %>%
  mutate(speaker_total_questions = sum(is_question)) %>%
  mutate(speaker_total_exclamations = sum(is_exclamation)) %>%
  mutate(speaker_total_statements = sum(is_statement)) %>%
  #repetitions
  mutate(speaker_total_repetitions = sum(repeat_nextTurn)) %>%
  ungroup() %>%
  #...............................................................
  #percentages
  mutate(speaker_percentage_Groupturns = speaker_total_Groupturns / total_Groupturns) %>%
  mutate(speaker_percentage_turns = speaker_total_turns / total_turns) %>%
  mutate(speaker_percentage_chars = speaker_total_chars / total_chars) %>%
  mutate(speaker_percentage_words = speaker_total_words / total_words) %>%
  #sentence types
  mutate(speaker_percentage_questionTags = speaker_total_questionTags / total_turns) %>%
  mutate(speaker_percentage_questions = speaker_total_questions / total_turns) %>%
  mutate(speaker_percentage_exclamations = speaker_total_exclamations / total_turns) %>%
  mutate(speaker_percentage_statements = speaker_total_statements / total_turns) %>%
  #repetitions
  mutate(speaker_percentage_repetitions = speaker_total_repetitions / total_turns) %>%
  #...............................................................
  #Intra speaker
  mutate(intra_speaker_percentage_questionTags = speaker_total_questionTags / speaker_total_turns) %>%
  mutate(intra_speaker_percentage_questions = speaker_total_questions / speaker_total_turns) %>%
  mutate(intra_speaker_percentage_exclamations = speaker_total_exclamations / speaker_total_turns) %>%
  mutate(intra_speaker_percentage_statements = speaker_total_statements / speaker_total_turns)
