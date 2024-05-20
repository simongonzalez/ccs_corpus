#https://drammock.github.io/phonR/

library(data.table)
library(tidyverse)
library(readxl)
library(emuR)
library(udpipe)
library(vowels)
library(joeyr)

`%nin%` <- Negate(`%in%`)



dat <- fread('df_phon.csv') %>%
  arrange(row_index)

dat_append <- fread('dat_append.csv') %>%
  arrange(row_index) %>%
  select(-one_of(c('row_index', 'speaker', 'time')))

df <- bind_cols(dat, dat_append) %>%
  mutate(duration_ms = dur * 1000) %>%
  filter(!grepl('^--', formant_1)) %>%
  filter(!grepl('^--', formant_2)) %>%
  filter(!grepl('^--', formant_3)) %>%
  mutate(formant_1 = as.numeric(formant_1), 
         formant_2 = as.numeric(formant_2), 
         formant_3 = as.numeric(formant_3)) %>%
  filter(segment_type_segments == 'vowel')

df %>%
  filter(onset >= 516.4515 & onset <= 516.4515) %>%
  View()

#=============================================
#Filter vowel spaces
# You can output the data to a column called something like "is_outlier" and
# then filter out values that are TRUE.
df_filter <- df %>%
  mutate(segmentid = paste0(segment)) %>%
  group_by(segmentid, speaker) %>%
  mutate(is_outlier = find_outliers(formant_1, formant_2, keep = 0.95)) %>%
  filter(!is_outlier)

 df_filter %>%
  filter(!is_outlier) %>%
  ggplot(aes(formant_2, formant_1, color = segment)) +
  geom_point() +
  scale_x_reverse(limits = c(3000, 500)) +
  scale_y_reverse(limits = c(1000, 200))

df_filtered <- df_filter %>%
  filter(!is_outlier)

df_filtered %>%
  ggplot(aes(formant_2, formant_1, color = segmentid)) +
  stat_ellipse(alpha = 0.5) +
  scale_x_reverse() +
  scale_y_reverse()

df2 <- df_filtered %>%
  mutate(F1 = formant_1) %>%
  mutate(F2 = formant_2) %>%
  mutate(gender = if_else(grepl('H', speaker), 'Male', 'Female'))

# dat_filtered <- sd_classify(data = df2, sd_value = 2, 
#                             f1_label = 'F1', 
#                             f2_label = 'F2', 
#                             vowel_column = 'segment',
#                             sex_column = 'gender',
#                             plot_vowels = T, compare = T)
# 
# 
# dat_filtered %>%
#   ggplot(aes(F2, F1, color = segmentid)) +
#   stat_ellipse(alpha = 0.5, level = 0.10) +
#   scale_x_reverse() +
#   scale_y_reverse()
# 
# df_filtered %>%
#   ggplot(aes(formant_2, formant_1, color = segmentid)) +
#   stat_ellipse(alpha = 0.5) +
#   scale_x_reverse() +
#   scale_y_reverse()# Alternatively, you can skip a step and just keep the data that are not
# # outliers.
# df %>%
#   group_by(vowel) %>%
#   filter(!find_outliers(F1, F2))

#=============================================
#Normalise vowels
dat_filtered <- df2
dat_filtered$context <- 1:nrow(dat_filtered)

#creates a data with the required columns to be input into the normalisation functions
dnorm <- dat_filtered[,c('speaker', 'segment', 'context', 'formant_1', 'formant_2', 'formant_3', 'formant_1', 'formant_2', 'formant_3'),]
names(dnorm) <- c('speaker_id', 'vowel_id', 'context', 'F1', 'F2', 'F3', 'F1_glide', 'F2_glide', 'F3_glide')
#Deletes the unnecessary columns
dnorm$F3 <- NA
dnorm$F1_glide <- NA
dnorm$F2_glide <- NA
dnorm$F3_glide <- NA

#Delete NA values
#dnorm <- dnorm[complete.cases(dnorm$F1, dnorm$F2),]

#Normalises formants using the Lobanov method
normed.vowels <- norm.lobanov(as.data.frame(dnorm))

#Rescales the values to Hz
rescaled <- scalevowels(normed.vowels)

#Renames columns to be compatible with the input data
names(normed.vowels) <- c('speaker', 'segment',       'context',    'F1_norm'    ,'F2_norm', 'F1b', 'F2b')
#Substes to the relevant columns
normed.vowels <- normed.vowels[,c('speaker', 'segment',       'context',    'F1_norm'    ,'F2_norm')]
normed.vowels <- normed.vowels %>%
  select(context, F1_norm, F2_norm)

#Renames columns to be compatible with the input data
names(rescaled) <- c('speaker', 'segment',       'context',    'F1_sc'    ,'F2_sc', 'F1b', 'F2b')
#Substes to the relevant columns
rescaled <- rescaled[,c('speaker', 'segment',       'context',    'F1_sc'    ,'F2_sc')]
rescaled <- rescaled %>%
  select(context, F1_sc, F2_sc)

#Merges the normalised values to the main data frame
dat_normalised <- dat_filtered %>%
  left_join(normed.vowels, by = 'context') %>%
  left_join(rescaled , by = 'context')

dat_normalised %>%
  ggplot(aes(F2_sc, F1_sc, color = segmentid, fill = segmentid)) +
  geom_point(alpha = 0.6) +
  stat_ellipse(alpha = 0.5, geom = 'polygon') +
  scale_x_reverse() +
  scale_y_reverse() +
  theme_minimal()

ggsave('vowelSpace.jpg', width = 25, height = 18, units = 'cm')

dat_normalised %>%
  group_by(segmentid) %>% 
  summarise(mean_dur_ms = mean(duration_ms)) %>%
  ggplot(aes(segmentid, mean_dur_ms, color = segmentid, fill = segmentid)) +
  geom_bar(stat='identity') +
  theme_minimal()

ggsave('durationPlots.jpg', width = 25, height = 18, units = 'cm')

metadata <- read.csv('metadata.csv')

dat_normalised_2 <- dat_normalised %>%
  filter(segment %in% c('a', 'A')) %>%
  dplyr::select(-one_of('gender')) %>%
  left_join(metadata, by = 'speaker') %>%
  mutate(corpus = factor(corpus))

dat_normalised_2 %>%
  filter(previous == 's') %>%
  ggplot(aes(F2_sc, F1_sc, color = interaction(gender, corpus, socialclass), 
             group = interaction(gender, corpus, socialclass))) +
  stat_ellipse(alpha = 0.5) +
  scale_x_reverse() +
  scale_y_reverse()

library(lmerTest)

mdl <- lmer(F1_sc ~ gender + corpus + socialclass + dur + speechRate + (1|speaker),
            data = dat_normalised_2 %>%
              filter(segment == 'A') %>%
              filter(previous == 's'))
mdl_sum <- summary(mdl)
mdl_sum

write.csv(dat_normalised_2, 'icphs_a_data.csv', row.names = F)

save(dat_normalised, file = '10_dat_normalised_20220817.RData')
save(dat_normalised_2, file = '10_dat_normalised_2_20220817.RData')
#write.csv(dat_normalised, '10_dat_normalised_20220817.csv', row.names = FALSE)
#...............................................................................

